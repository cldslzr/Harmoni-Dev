# HARMONI — Context-Aware Conversation

A framework for context-aware, multi-turn conversations with robots. HARMONI enables embodied conversational agents to dynamically adapt to individuals, conversational contexts, and interaction scenarios, combining perception, user modeling, memory retrieval, and response generation.

## Overview

HARMONI is designed to support natural, context-aware interactions between humans and robots. It combines multimodal perception, user identification, contextual and temporal memory, and language generation to provide responses adapted to the ongoing interaction.

The framework has been updated to improve responsiveness, flexibility, memory retrieval, and ease of deployment, while supporting both live multimodal interaction and manually provided audio/video inputs.

![System Architecture](documentation/architecture.png)

## Main Features

- **Multimodal Perception:** Processes audio and video to transcribe speech, detect faces, extract facial embeddings, and identify the active speaker.
- **User Modeling:** Retrieves and maintains user-specific profiles and long-term memory using facial embeddings.
- **Contextual Memory:** Maintains conversational context and retrieves relevant information from past interactions.
- **Temporal Awareness:** Enables the system to answer time-related questions based on stored information, such as meeting dates, past events, or how a trip went.
- **Context Adaptation:** Dynamically adapts the system to different interaction scenarios through configurable contexts, such as a reception robot or a hospital kiosk.
- **Knowledge-Rich Retrieval:** Uses Retrieval-Augmented Generation (RAG) to retrieve information from large amounts of contextual data and answer complex, knowledge-intensive questions efficiently.
- **Multimodal Interaction:** Supports interactions through local cameras and microphones, as well as manually provided audio and video files.
- **Parallel Processing:** Runs perception tasks, including audio transcription, face detection, and other processing steps, in parallel to reduce processing time.
- **Optimized Model Initialization:** Loads and warms up models before the first interaction to reduce initial response latency.
- **Flexible LLM Selection:** Supports experimentation with different language models to find a balance between response quality and inference speed.

## Work Interfaces

HARMONI provides interfaces to test and interact with the system.

### Video Interaction

The video interaction interface supports multimodal interaction with HARMONI through:

- Live video and audio input from a local camera and microphone.
- Manual submission of audio and video files for processing.
- Speech transcription and active speaker identification.
- User identification and retrieval of associated memory.
- Contextual response generation.

The perception and processing pipeline has been streamlined to reduce unnecessary computation and improve responsiveness.

## Recent Improvements

### General System Improvements

- **Removal of the Web API:** Removed the previous Web API component.
- **Model Warmup:** Models are loaded and warmed up before the first query, allowing the system to be ready for use and reducing latency during the first interaction.
- **Parallel Processing:** Parallelized processing tasks, including audio transcription, face detection, and other perception components, to reduce overall processing time.
- **Temporal Awareness:** Added temporal reasoning capabilities to support questions about events, dates, meetings, trips, and other time-related information.
- **Dynamic Context Management:** Introduced a context system allowing HARMONI to be configured for different interaction scenarios and environments.
- **RAG Integration:** Added Retrieval-Augmented Generation to handle large amounts of contextual information and answer complex, knowledge-rich questions more efficiently.
- **Multimodal Input:** Added support for local PC camera and microphone input, alongside manually provided audio and video files.
- **LLM Benchmarking:** Evaluated different language models to identify suitable models balancing response quality and inference speed.

### Vision Pipeline Improvements

The vision pipeline has been simplified and optimized, particularly regarding face detection, embedding extraction, and user retrieval.

- **InsightFace Integration:** Replaced the previous model usage with the `FaceAnalysis` interface from InsightFace, allowing face detection and embedding extraction to be handled directly through the same interface.
- **Model Initialization:** Updated InsightFace initialization in `config/models.py`, specifying providers and allowed modules.
- **Face-Based User Retrieval:** Simplified user retrieval in `conversation/memory/memory.py` to accept face embeddings directly, avoiding unnecessary image processing.
- **Video Processing:** Updated `video_interface/app.py` to handle embeddings directly during user retrieval.
- **Simplified Face Detection:** Removed the previous grid-based detection approach and the additional YOLO-based face detection step.
- **Code Cleanup:** Removed unused utility functions from `vision/utils.py` and streamlined face detection in `vision/detection.py`.
- **Improved Speaking Face Identification:** Simplified the active speaker identification logic to return face embeddings and images directly.
- **Configurable Thresholds:** Added and defined thresholds in configuration and settings files to better control system behavior.
- **Model Warmup:** Updated the warmup procedure to use an image containing a face rather than a blank image, ensuring that the face-related processing pipeline is properly initialized. A new test image (`test.png`) has been added to `video_interface` for this purpose.

> **Note:** Emotion detection is currently disabled in the vision pipeline.

## Installation

### Requirements

- [Docker](https://docs.docker.com/engine/install/)
- NVIDIA GPU and drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 1. Install Docker

Follow the official instructions for your operating system: https://docs.docker.com/engine/install/

On Arch Linux:

```bash
sudo pacman -S docker
sudo systemctl enable --now docker
```

### 2. Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker containers to access your GPU.

On Arch Linux:

```bash
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### 3. Clone the Repository

```bash
git clone git@github.com:MalecotJeanne/Context-aware_Conversation.git
cd Context-aware_Conversation
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root containing the required API keys:

```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

> **Warning:** Never commit your `.env` file or expose your API keys in version control.

### 5. Build the Docker Image

```bash
docker build --build-arg HF_TOKEN=your_token_here -t harmoni:latest .
```

The first build may take several minutes while installing dependencies. Subsequent builds may be faster due to Docker layer caching.

### 6. Run the Container

```bash
docker run -p 8000:8000 --runtime=nvidia --gpus all --env-file .env -v "$PWD":/app harmoni:latest
```

Once the container is running, open the interface in your browser:

**http://localhost:8000**

| Flag | Description |
|---|---|
| `-p 8000:8000` | Maps the container port 8000 to port 8000 on the host machine. |
| `--runtime=nvidia` | Uses the NVIDIA container runtime. |
| `--gpus all` | Provides the container with access to all available GPUs. |
| `--env-file .env` | Loads environment variables from the `.env` file. |
| `-v "$PWD":/app` | Mounts the repository into the container. |

## Configuration

HARMONI can be adapted to different scenarios through its configuration files.

### Settings

The [`config/settings.py`](./config/settings.py) file contains configurable parameters controlling system behavior, including thresholds and other pipeline settings.

Changes should be made carefully, as improper settings may affect system performance.

### Context and Models

The configuration files in the `config` directory allow the system to be adapted to different scenarios and to select the models used throughout the pipeline.

The models used for perception and generation can be modified to experiment with different approaches and trade-offs between speed and quality.

## Future Improvements

Potential future improvements include:

- Maintaining multiple face embeddings per person to improve identification robustness.
- Regularly updating stored embeddings as new interactions occur.
- Selecting or maintaining the most representative embedding for each person over time.
