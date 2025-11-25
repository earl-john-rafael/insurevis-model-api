# Use a lightweight official Python image
FROM python:3.9-slim

# --- 1. Install system dependencies ---
# Essential for OpenCV (libgl1) and compiling Detectron2/PyTorch extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /insurevis

# --- 2. Install Python Dependencies ---
COPY requirements.txt .

# Install CPU-only PyTorch first to keep image size small (Cloud Run uses CPU by default)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- 3. Copy Application Code ---
# This copies app.py, config.json, etc.
COPY . .

# --- 4. Setup Model Directory ---
# We create an EMPTY folder here. 
# When you deploy, Google Cloud will "inject" your bucket files into this folder.
RUN mkdir -p /models

# --- 5. Configure Port & Startup ---
# Cloud Run sends requests to port 8080 by default.
ENV PORT=8080
EXPOSE 8080

# Start the server using Gunicorn
# --workers 1: Limits memory usage (good for standard Cloud Run instances)
# --threads 8: Allows handling multiple requests at once
# --timeout 0: Disables Gunicorn timeout (lets Cloud Run handle it)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app