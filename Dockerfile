# Use a lightweight official Python image
FROM python:3.9-slim

# --- 1. Install system dependencies ---
# 'build-essential' and 'git' are required to compile Detectron2
# 'libgl1' and 'libglib2.0-0' are required for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /insurevis

# --- 2. Install Dependencies ---
COPY requirements.txt .

# STEP A: Install CPU-only PyTorch first.
# We do this separately to ensure we don't accidentally download the massive GPU version.
# This keeps your container small and fast.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# STEP B: Install the rest of the requirements (Detectron2, Flask, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# --- 3. Copy Application Code ---
COPY . .

# --- 4. Setup Model Directory ---
# Create the directory where Google Cloud will mount the bucket (or where you put local models)
RUN mkdir -p /models

# --- 5. Configure Port & Startup ---
ENV PORT=8080
EXPOSE 8080

# STARTUP COMMAND (Optimized for 6-7 Users)
# --preload:    Loads the models ONCE in memory, then shares that memory across workers. 
#               CRITICAL for running Detectron2 on limited RAM.
# --workers 3:  Creates 3 parallel processes. This allows 3 users to be processed 
#               at the EXACT same time.
# --threads 4:  Allows each worker to handle overlapping requests.
# --timeout 120: Gives heavy image processing time to finish before crashing.
CMD exec gunicorn --bind :$PORT --workers 3 --threads 4 --timeout 120 --preload app:app