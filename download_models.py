"""
Pre-downloads all AI models into the Docker image cache so containers
don't need network access at startup.
Run during docker build (no GPU required — models are loaded on CPU).
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import whisper
from transformers import AutoImageProcessor, AutoModelForImageClassification
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as insightface_get_model

# 1. Whisper
print("Downloading Whisper 'small' model...")
whisper.load_model("small")
print("Whisper done.")

# 2. HuggingFace facial emotion model
print("Downloading facial emotion model...")
AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection", use_fast=True)
AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
print("Emotion model done.")

# 3. InsightFace buffalo_l
print("Downloading InsightFace buffalo_l model...")
model = insightface_get_model("buffalo_l", download=True)
if model is None:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)  # -1 = CPU
else:
    model.prepare(ctx_id=-1)
print("InsightFace done.")

print("All models downloaded successfully.")