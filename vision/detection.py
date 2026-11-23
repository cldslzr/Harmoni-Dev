"""
This module provides functionality for face detection and speaker recognition in video frames.
"""

#--------------------------------------- Imports ---------------------------------------#

import os
import sys
import time

import cv2
import numpy as np
from moviepy import VideoFileClip
from PIL import Image
import torch

from insightface.app import FaceAnalysis

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI warnings
from sklearn.metrics.pairwise import cosine_similarity

from config.utils import suppress_stdout
from config.settings import FRAME_STRIDE, FILTER_THRESHOLD

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from vision.audio import extract_mono_wav

#--------------------------------------- Initialization ---------------------------------------#

with suppress_stdout():
    INSIGHTFACE_MODEL = FaceAnalysis(
    name='buffalo_l', 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'landmark_2d_106', 'recognition']
)
    INSIGHTFACE_MODEL.prepare(ctx_id=0, det_size=(640, 640))
    

#--------------------------------------- Functions ---------------------------------------#

def find_person_id(face_embedding, known_faces, threshold=0.5):
    if not known_faces:
        return None

    if isinstance(face_embedding, np.ndarray):
        face_embedding = torch.tensor(face_embedding, dtype=torch.float32)

    person_ids = list(known_faces.keys())
    known_embeddings = torch.stack([
        torch.tensor(e, dtype=torch.float32) if isinstance(e, np.ndarray) else e
        for e in known_faces.values()
    ])

    cos = torch.nn.CosineSimilarity(dim=-1)
    similarities = cos(face_embedding.unsqueeze(0), known_embeddings)

    best_score, best_idx = torch.max(similarities, dim=0)

    if best_score.item() > threshold:
        return person_ids[best_idx.item()]

    return None


def detect_speaking_face(video_path):
    """
    Detect the speaking face in a video stream using FaceAnalysis from InsightFace.
    Args:
        video_path: Path to the video file.
        audio_path: Optional path to a separate audio file.
    Returns:
        np.ndarray: Embedding of the speaking person, or None if no face detected.
        PIL.Image: Best cropped image of the speaking person's face, or None.
    """
    known_faces = {}       # person_id -> embedding
    person_landmarks = {}  # person_id -> list of landmark_2d_106
    person_crops = {}      # person_id -> list of (det_score, cropped image)
    next_person_id = 0
    i = 0

    # Read frames from the video
    cap = cv2.VideoCapture(video_path)
    cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:  # ret false for last frame
            break

        if i % FRAME_STRIDE == 0:
            faces = INSIGHTFACE_MODEL.get(frame)

            for face in faces:

                # Look for existing person
                person_id = find_person_id(
                    face.embedding,
                    known_faces,
                    threshold=0.5
                )
                

                # New person
                if person_id is None:
                    person_id = next_person_id
                    known_faces[person_id] = face.embedding
                    next_person_id += 1
                    person_landmarks[person_id] = []
                    person_crops[person_id] = []

                # Crop
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                person_landmarks[person_id].append(face.landmark_2d_106)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                person_crops[person_id].append((face.det_score, Image.fromarray(crop_rgb), face.embedding))

        i += 1

    # Release the video capture object
    cap.release()

    # Filter out persons with too few detections (outliers)
    valid_ids = [pid for pid, lms in person_landmarks.items() if len(lms) >= FILTER_THRESHOLD]

    if not valid_ids:
        return None, None

    # Compute speaking probability for each detected person
    probs = [compute_speaking_probability(person_landmarks[pid]) for pid in valid_ids]

    best_idx = np.argmax(probs)
    speaking_id = valid_ids[best_idx]

    # Pick the best-quality frame (highest detection score) for the speaker
    _, best_image, best_embedding = max(person_crops[speaking_id],key=lambda x: x[0])

    # Return  all images and one specific user embeddings if needed
    # all_images = [img for _, img in person_crops[speaking_id]]
    # return known_faces[speaking_id], all_images

    return best_embedding, best_image

def compute_speaking_probability(landmark_sequences, smooth_window=3, threshold=0.02):
    """
    Estimate speaking probability using MAR (Mouth Aspect Ratio) combined with
    eye openness and gaze alignment, from InsightFace's 106-point 2D landmarks.

    Parameters:
        landmark_sequences: list or np.ndarray of shape [T, 106, 2]
            - T = number of frames where this person was detected
            - Each element must be the FULL 106-point landmark array
              (not a lips-only subset), since eye/pupil indices are also used.
        smooth_window: int
            - Rolling window size for temporal smoothing of the MAR signal
        threshold: float
            - Sensitivity for MAR change above the estimated baseline (closed mouth)

    Returns:
        float: Mean speaking probability across all frames for this person.
    """
    landmarks = np.array(landmark_sequences)  # [T, 106, 2]
    T = landmarks.shape[0]

    if T == 0:
        return 0.0

    mar_values = []

    for t in range(T):
        ld = landmarks[t]

        # Distance between the eyes (normalization reference)
        eye_dist = np.linalg.norm(ld[39] - ld[89])

        # Inner distances between lips
        D_in_c = np.linalg.norm(ld[62] - ld[60])   # center of the mouth
        D_in_l = np.linalg.norm(ld[66] - ld[54])   # left side of the mouth
        D_in_r = np.linalg.norm(ld[70] - ld[57])   # right side of the mouth

        A = (D_in_c + D_in_r + D_in_l) / 3.0  # Average inner distance

        # Distance between mouth corners
        d_corners = np.linalg.norm(ld[52] - ld[61])
        normalized_d_corners = d_corners / (eye_dist + 1e-6)

        mar = A / (normalized_d_corners + 1e-6)
        mar_values.append(mar)

    mar_values = np.array(mar_values)

    # Smooth over time
    if smooth_window > 1 and T >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        mar_smooth = np.convolve(mar_values, kernel, mode='same')
    else:
        mar_smooth = mar_values

    # Estimate baseline (closed mouth MAR)
    baseline = np.percentile(mar_smooth, 10)
    mar_delta = mar_smooth - baseline

    # Feature likelihood from mouth movement
    p_vis = 1 / (1 + np.exp(-(mar_delta / threshold)))

    # --- Pupil visibility / gaze alignment ---
    right_open, left_open = [], []
    right_gaze, left_gaze = [], []

    for t in range(T):
        ld = landmarks[t]

        # Right eye
        top_r, bottom_r = ld[94], ld[87]
        left_r, right_r = ld[89], ld[93]
        pupil_r = ld[88]
        width_r = np.linalg.norm(left_r - right_r)
        open_r = np.linalg.norm(top_r - bottom_r) / (width_r + 1e-6)
        center_r = (left_r + right_r) / 2
        gaze_r = np.linalg.norm(pupil_r - center_r) / (width_r + 1e-6)

        # Left eye
        top_l, bottom_l = ld[40], ld[33]
        left_l, right_l = ld[35], ld[39]
        pupil_l = ld[38]
        width_l = np.linalg.norm(left_l - right_l)
        open_l = np.linalg.norm(top_l - bottom_l) / (width_l + 1e-6)
        center_l = (left_l + right_l) / 2
        gaze_l = np.linalg.norm(pupil_l - center_l) / (width_l + 1e-6)

        right_open.append(open_r)
        left_open.append(open_l)
        right_gaze.append(gaze_r)
        left_gaze.append(gaze_l)

    right_open, left_open = np.array(right_open), np.array(left_open)
    right_gaze, left_gaze = np.array(right_gaze), np.array(left_gaze)

    # Eye openness (probability eye is open)
    eye_open = (right_open + left_open) / 2
    p_open = 1 / (1 + np.exp(-(eye_open - 0.25) * 20))

    # Gaze alignment (probability looking at center)
    gaze_offset = (right_gaze + left_gaze) / 2
    p_gaze = np.exp(-(gaze_offset ** 2) / (0.05 ** 2))

    # Final probability: open eyes AND looking at center
    p_pupil = p_open * (1 - p_gaze)

    # Combine mouth movement and gaze/eye probabilities
    alpha = 0.75
    prob = alpha * p_vis + (1 - alpha) * p_pupil

    return float(np.mean(prob))