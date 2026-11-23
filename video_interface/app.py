"""
# app.py
This file contains the main FastAPI application for handling user interactions, and video processing, 
including detecting speaking user, retrieving user profiles, answering questions, and updating user memory.
It integrates with the memory module to store and retrieve user-specific data.
It also handles user image for face recognition and user identification.
==========
Run the application in the root dir using:
    uvicorn video_interface.app:app --reload
"""

#--------------------------------------- Imports ---------------------------------------#

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
warnings.filterwarnings("ignore", "pkg_resources is deprecated", category=UserWarning)

import asyncio
import queue
import threading
import time
import tempfile
import shutil
import sqlite3
import json
import io
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import yaml

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from config import models, get_face_embedding_model
from config.settings import LEN_HISTORY

from conversation.llm.openai_inferences import generate_answer, generate_answer_stream, update_memory_llm
from conversation.llm.retriever import context_files_retriever, get_context_file_index
from conversation.memory.memory import user_retriever, update_memory
from conversation.memory.utils import (
    add_conversation_turn,
    clear_conversation_history,
    create_table,
    empty_database,
    get_conversation_history,
    timestamp_conversation_content,
)

from vision.detection import detect_speaking_face
from vision.audio import extract_audio, transcribe_audio
from vision.emotions import detect_emotions

import cv2

#--------------------------------------- Configuration ---------------------------------------#

@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_models()
    yield

# Init FastAPI app
app = FastAPI(lifespan=lifespan)

# Global variables for user session management
# Load models from vision module
stt_model = models.WHISPER_MODEL
emotion_model = models.EMOTION_MODEL
emotion_processor = models.EMOTION_PROCESSOR

# Load user retriever model and processor
user_retriever_config = get_face_embedding_model("INSIGHTFACE")

# Database connection for user retrieval
database = 'video_interface/users.db'
conn = sqlite3.connect(database)
create_table(conn)   
conn.close() 

current_user_image = None
memory_user = None
current_context = ""  # contexte statique optionnel (peut être surchargé via /set_context)

config_path = "config/config.yaml"
with open(config_path, "r", encoding="utf-8") as _f:
    config_data = yaml.safe_load(_f)

# Pré-calcul des embeddings des fichiers de contexte au démarrage
get_context_file_index(config_data)


#--------------------------------------- Model Warmup ---------------------------------------#

def warmup_models():
    """Run dummy inference through all models to force GPU memory allocation."""
    import torch
    print("[Warmup] Pre-loading all models into VRAM...", flush=True)
    start = time.time()

    # 1. InsightFace face detection (INSIGHTFACE_MODEL in vision/detection.py)
    from vision.detection import INSIGHTFACE_MODEL
    test_path = Path(__file__).resolve().parent / "test.png"
    print(f"Loading {test_path}", flush=True)
    test_img = cv2.imread(str(test_path))

    assert test_img is not None, "Impossible de charger test.png"
    INSIGHTFACE_MODEL.get(test_img)
    print(f"  [Warmup] INSIGHTFACE_MODEL (face detection, face embeddings) ready", flush=True)

    # 2. Whisper (transcription)
    import whisper
    dummy_audio = whisper.pad_or_trim(torch.zeros(16000, dtype=torch.float32))
    mel = whisper.log_mel_spectrogram(dummy_audio).to(stt_model.device)
    stt_model.decode(mel.unsqueeze(0), whisper.DecodingOptions(fp16=False))
    print(f"  [Warmup] Whisper (transcription) ready", flush=True)

    # 3. Emotion model
    dummy_pil = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    inputs = emotion_processor(images=dummy_pil, return_tensors="pt")
    device = next(emotion_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        emotion_model(**inputs)
    print(f"  [Warmup] Emotion model ready", flush=True)

    elapsed = time.time() - start
    print(f"[Warmup] All models loaded into VRAM in {elapsed:.2f}s", flush=True)


#--------------------------------------- Routes ---------------------------------------#

@app.get("/config")
def get_config():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config 


@app.post("/process")
async def process_video(video: UploadFile = File(...), audio: UploadFile = File(None)):
    """
    Handle the user's video input, detect the speaking user, transcribe speech,
    detect emotions, retrieve user profile, generate an answer, and update memory.
    If a separate audio file is provided, use it directly; otherwise extract from video.
    """
    print("PROCESS ENDPOINT ENTRANCE")
    endpoint_start_time = time.time()

    global current_user_image, memory_user

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    if audio:
        # Audio provided separately — run face detection and transcription in parallel
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            audio_file = tmp_audio.name

        with ThreadPoolExecutor(max_workers=2) as executor:
            detectface_start_time = time.time()
            future_face = executor.submit(detect_speaking_face, tmp_path)

            tr_start_time = time.time()
            future_transcript = executor.submit(transcribe_audio, audio_file, stt_model)

            speaking_face_embedding, speaking_face_image = future_face.result()
            print(f"DETECT SPEAKING FACE EXECUTION TIME : {time.time() - detectface_start_time:.4f} seconds")

            transcript = future_transcript.result()
            print(f"TRANSCRIPTION EXECUTION TIME : {time.time() - tr_start_time:.4f} seconds")

        os.remove(audio_file)
    else:
        # No separate audio — extract from video
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_audio_extract = executor.submit(extract_audio, tmp_path, audio_file)

            detectface_start_time = time.time()
            speaking_face_embedding, speaking_face_image = detect_speaking_face(tmp_path)
            print(f"DETECT SPEAKING FACE EXECUTION TIME : {time.time() - detectface_start_time:.4f} seconds")

            future_audio_extract.result()  # Ensure extraction is done

        tr_start_time = time.time()
        transcript = transcribe_audio(audio_file, stt_model)
        print(f"TRANSCRIPTION EXECUTION TIME : {time.time() - tr_start_time:.4f} seconds")
        os.remove(audio_file)

    if speaking_face_embedding is None or speaking_face_image is None:
        raise HTTPException(status_code=422, detail="No face detected in the video.")

    transcript_timestamp = datetime.now()
    timestamped_transcript = timestamp_conversation_content(transcript, transcript_timestamp)


    with ThreadPoolExecutor(max_workers=2) as executor:

        # TODO: Update emotion detection
        # emo_start_time = time.time()
        # future_emotion = executor.submit(detect_emotions, speaking_face_row, emotion_model, emotion_processor)

        ur_start_time = time.time()
        future_user = executor.submit(user_retriever, speaking_face_embedding, None, database)

        # emotion, prob, emotions, probs = future_emotion.result()
        # print(f"EMOTION DETECTION EXECUTION TIME : {time.time() - emo_start_time:.4f} seconds")

        detected_user, memory_user = future_user.result()
        print(f"USER RETRIEVER EXECUTION TIME : {time.time() - ur_start_time:.4f} seconds")


    conn_history = sqlite3.connect(database)
    try:
        session_history = get_conversation_history(detected_user, conn_history, LEN_HISTORY)
    finally:
        conn_history.close()

    # Start memory update thread immediately once transcript and detected_user are available
    def update_user_memory():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)
        try:
            new_memory_object = update_memory_llm(timestamped_transcript, conn_thread, detected_user, stm=session_history)
            global memory_user
            memory_user = update_memory(new_memory_object, detected_user, conn_thread)
        finally:
            conn_thread.close()
        print(f"MEMORY UPDATE TIME : {time.time() - start_time:.2f} seconds")

    memory_thread = threading.Thread(target=update_user_memory, daemon=True)
    memory_thread.start()

    visual_profile = {
        "emotion": None,
        "gender": None,
        "age": None,
    }

    # Dynamically retrieves the relevant context files for this question
    dynamic_context = context_files_retriever(transcript, config_data)
    combined_context = "\n\n".join(filter(None, [current_context, dynamic_context]))

    print(f"COMBINED CONTEXT:\n{combined_context}")

    # Run generate_answer in background while building HTML blocks and encoding the image
    def _run_generate_answer():
        _conn = sqlite3.connect(database)
        try:
            t = time.time()
            result = generate_answer(timestamped_transcript, session_history, combined_context, _conn, detected_user, visual_profile)
            return result, time.time() - t
        finally:
            _conn.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_answer = executor.submit(_run_generate_answer)

        (answer, retrieved_features), llm_time = future_answer.result()

    conn_history = sqlite3.connect(database)
    try:
        add_conversation_turn(detected_user, transcript, answer, conn_history, user_timestamp=transcript_timestamp)
    finally:
        conn_history.close()

    print("PROCESS ENDPOINT DONE (memory update continues in background)")
    print(f"PROCESS ENDPOINT EXECUTION TIME : {time.time() - endpoint_start_time:.4f} seconds")

    return {
        "detected_user": detected_user,
        "profile": memory_user,
        # "emotion": emotion,
        "transcript": transcript,
        "answer": answer,
    }

@app.post("/process_stream")
async def process_video_stream(video: UploadFile = File(...), audio: UploadFile = File(None)):
    """
    Streaming version of /process for terminal clients.
    Yields newline-delimited JSON events:
      {"type": "info", "detected_user": "...", "emotion": "...", "transcript": "..."}
      {"type": "token", "content": "..."}  (one per LLM token)
      {"type": "done"}
    """
    print("PROCESS_STREAM ENDPOINT ENTRANCE")
    endpoint_start_time = time.time()

    global current_user_image, memory_user

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            audio_file = tmp_audio.name

        with ThreadPoolExecutor(max_workers=2) as executor:
            detectface_start_time = time.time()
            future_face = executor.submit(detect_speaking_face, tmp_path, save_frames=True)

            tr_start_time = time.time()
            future_transcript = executor.submit(transcribe_audio, audio_file, stt_model)

            speaking_face_row, _, _ = future_face.result()
            print(f"DETECT SPEAKING FACE EXECUTION TIME : {time.time() - detectface_start_time:.4f} seconds")

            transcript = future_transcript.result()
            print(f"TRANSCRIPTION EXECUTION TIME : {time.time() - tr_start_time:.4f} seconds")

        os.remove(audio_file)
    else:
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_audio_extract = executor.submit(extract_audio, tmp_path, audio_file)

            detectface_start_time = time.time()
            speaking_face_row, _, _ = detect_speaking_face(tmp_path, save_frames=True)
            print(f"DETECT SPEAKING FACE EXECUTION TIME : {time.time() - detectface_start_time:.4f} seconds")

            future_audio_extract.result()

        tr_start_time = time.time()
        transcript = transcribe_audio(audio_file, stt_model)
        print(f"TRANSCRIPTION EXECUTION TIME : {time.time() - tr_start_time:.4f} seconds")
        os.remove(audio_file)

    if not speaking_face_row:
        raise HTTPException(status_code=422, detail="No face detected in the video.")

    transcript_timestamp = datetime.now()
    timestamped_transcript = timestamp_conversation_content(transcript, transcript_timestamp)

    user_image = speaking_face_row[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        emo_start_time = time.time()
        future_emotion = executor.submit(detect_emotions, speaking_face_row, emotion_model, emotion_processor)

        ur_start_time = time.time()
        future_user = executor.submit(user_retriever, user_image, None, user_retriever_config, database)

        emotion, _, _, _ = future_emotion.result()
        print(f"EMOTION DETECTION EXECUTION TIME : {time.time() - emo_start_time:.4f} seconds")

        detected_user, memory_user = future_user.result()
        print(f"USER RETRIEVER EXECUTION TIME : {time.time() - ur_start_time:.4f} seconds")

    current_user_image = np.array(user_image)

    conn_history = sqlite3.connect(database)
    try:
        session_history = get_conversation_history(detected_user, conn_history, LEN_HISTORY)
    finally:
        conn_history.close()

    def update_user_memory_stream():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)
        try:
            new_memory_object = update_memory_llm(timestamped_transcript, conn_thread, detected_user, stm=session_history)
            global memory_user
            memory_user = update_memory(new_memory_object, detected_user, conn_thread)
        finally:
            conn_thread.close()
        print(f"MEMORY UPDATE TIME : {time.time() - start_time:.2f} seconds")

    threading.Thread(target=update_user_memory_stream, daemon=True).start()

    visual_profile = {"emotion": emotion, "gender": None, "age": None}
    dynamic_context = context_files_retriever(transcript, config_data)
    combined_context = "\n\n".join(filter(None, [current_context, dynamic_context]))

    print(f"COMBINED CONTEXT:\n{combined_context}")

    async def event_generator():
        yield json.dumps({
            "type": "info",
            "detected_user": detected_user,
            "emotion": emotion,
            "transcript": transcript,
        }) + "\n"

        token_queue = queue.Queue()
        answer_parts = []

        def stream_answer():
            _conn = sqlite3.connect(database)
            try:
                _, stream = generate_answer_stream(
                    timestamped_transcript, session_history, combined_context,
                    _conn, detected_user, visual_profile
                )
                generation_start_time = time.time()
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        answer_parts.append(token)
                        token_queue.put(token)
                print(f"GENERATION TIME : {time.time() - generation_start_time:.2f} seconds")
            except Exception as e:
                print(f"[process_stream] Streaming error: {e}")
            finally:
                _conn.close()
                token_queue.put(None)  # sentinel

        threading.Thread(target=stream_answer, daemon=True).start()

        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, token_queue.get)
            if item is None:
                break
            yield json.dumps({"type": "token", "content": item}) + "\n"

        full_answer = "".join(answer_parts)
        conn_history = sqlite3.connect(database)
        try:
            add_conversation_turn(detected_user, transcript, full_answer, conn_history, user_timestamp=transcript_timestamp)
        finally:
            conn_history.close()

        print("PROCESS_STREAM ENDPOINT DONE (memory update continues in background)")
        print(f"PROCESS_STREAM ENDPOINT EXECUTION TIME : {time.time() - endpoint_start_time:.4f} seconds")

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/reset_session")
async def reset_session(user_id: str = Form(None)):
    global current_user_image
    current_user_image = None
    if user_id:
        conn = sqlite3.connect(database)
        try:
            clear_conversation_history(user_id, conn)
        finally:
            conn.close()
    return {"status": "success"}


@app.post("/reset_database")
async def reset_database():
    conn = sqlite3.connect(database)

    empty_database(conn)

    conn = sqlite3.connect(database) 
    create_table(conn) 

    conn.close()
    return {"status": "success"}

@app.get("/get_context")
async def get_context():
    """
    Get the current context for the user.
    """
    global current_context
    return {"context": current_context}

@app.post("/set_context")
async def set_context(context: str = Form(...)):
    """
    Update the context for the user.
    """
    global current_context
    current_context = context
    return {"status": "success"}

@app.post("/identify_user")
async def identify_user(user_image: UploadFile = File(...)):
    """
    Edit user information.
    """
    global current_user_image
    image_bytes = await user_image.read()

    current_user_image = Image.open(io.BytesIO(image_bytes))
    current_user_image = current_user_image.convert("RGB")
    current_user_image = np.array(current_user_image)

    detected_user, memory_user = user_retriever(current_user_image, None, user_retriever_config, database)
    
    return {
        "user": detected_user,
        "profile": memory_user
    }

@app.post("/edit_user")
async def edit_user(
    user_id: str = Form(...),
    profile_data: str = Form(...),
    ):
    """
    Edit user information in the database.
    """
    # Parse the JSON string sent from the frontend
    try:
        profile_dict = json.loads(profile_data)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON data: {str(e)}"}

    # update the user profile in the database here
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    # reset the user table
    cursor.execute(f"DELETE FROM {user_id}")
    conn.commit()

    # Insert the new profile data
    for feature in profile_dict:
        values_list = feature.get("value", [])
        value = ';'.join(values_list) if isinstance(values_list, list) else values_list
        cursor.execute(
            f"INSERT INTO {user_id} (type, name, description, value) VALUES (?, ?, ?, ?)",
            (feature.get("type"), feature.get("name"), feature.get("description"), value),
        )

    conn.commit()
    conn.close()
    return {"status": "success"}
    
    
