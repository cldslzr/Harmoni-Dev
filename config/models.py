"""
This module contains the configuration for various vision models used in the application.
It includes models for face detection, landmark detection, audio transcription, emotion detection, and user retrieval
"""
#--------------------------------------- Imports ---------------------------------------#

import sys
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

import dotenv
import whisper

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from insightface.app  import FaceAnalysis
from config.utils import suppress_stdout

from config.settings import DEVICE

################
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#--------------------------------------- Configuration ---------------------------------------#

# Load Whisper model
WHISPER_MODEL = whisper.load_model("small").to(DEVICE)  # or parameterize this

# Load facial emotion model
#EMOTION_PROCESSOR = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
EMOTION_PROCESSOR = AutoImageProcessor.from_pretrained(
    "dima806/facial_emotions_image_detection", 
    use_fast=True
)
EMOTION_MODEL = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(DEVICE)

###################
# FACE EMBEDDINGS
###################

# INSIGHTFACE
with suppress_stdout():
    INSIGHTFACE_MODEL = FaceAnalysis(
    name='buffalo_l',  # ou le nom de pack que tu utilises déjà
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'landmark_2d_106', 'recognition']
)
    INSIGHTFACE_MODEL.prepare(ctx_id=0, det_size=(640, 640))