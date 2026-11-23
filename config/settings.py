"""
This module contains the configuration settings for the vision module. 
"""

#--------------------------------------- Imports ---------------------------------------#

import os
from dotenv import load_dotenv
import torch

#--------------------------------------- Configuration ---------------------------------------#

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable.")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Please set the OPENAI_API_KEY environment variable if you want to use OpenAI models.")

# Device setup
#DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

##########
# Video settings
##########

# Frame buffer size
LEN_FRAME_BUFFER = 5
FRAME_STRIDE = 15 # Process every n frame to reduce computation

##########
# Conversation settings
##########

LEN_HISTORY = 10
RECOGNITION_THRESHOLD = 0.85  # Distance threshold for user retrieval
NUM_TOP_FEATURES = 10 # Number of top features to retrieve for context

##########
# Face detection settings
##########

FILTER_THRESHOLD = 2

##########
# User retriever settings
##########

SIMILARITY_THERESHOLD = 0.4