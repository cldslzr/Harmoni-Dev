FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04

WORKDIR /app

ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    cmake \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python and install pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3.12 -m ensurepip --upgrade && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install openai-clip av dlib
RUN pip install -r requirements.txt
RUN pip uninstall onnx onnxruntime -y
RUN pip install onnx onnxruntime-gpu kokoro sounddevice

# Pre-download AI models into the image
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
COPY download_models.py .
RUN python download_models.py

COPY . .

EXPOSE 8000

CMD sh -c '\
    if command -v nvidia-smi >/dev/null 2>&1; then \
        nvidia-smi || true; \
    fi; \
    exec uvicorn video_interface.app:app \
        --reload \
        --host 0.0.0.0 \
        --port 8000'