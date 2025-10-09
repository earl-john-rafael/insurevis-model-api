# Use a lightweight official Python image
FROM python:3.9-slim

# --- Install system dependencies for Detectron2 and OpenCV ---
# This "toolbox" is essential for compiling complex libraries.
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

# Set the working directory inside the container
WORKDIR /insurevis

# Copy the requirements file and install Python packages
# This step will now succeed because the build tools are present.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application source code (e.g., app.py)
COPY . .

# --- Download Your Four Models from Azure Blob Storage ---
# Create the absolute target directory expected by the app (/models)
RUN mkdir -p /models

# !!! CRUCIAL STEP: Replace the placeholder URLs below with your actual model URLs !!!
# Use one 'ADD' line for each of your four models. Destination paths must exactly match
# the filenames referenced in `app.py` (see MODEL_DIR constants).
ADD ["https://insurevisstorage.blob.core.windows.net/models/Car Damage Segmentation Model.pth","/models/Car Damage Segmentation Model.pth"]
ADD ["https://insurevisstorage.blob.core.windows.net/models/Car Parts Segmentation Model.pth","/models/Car Parts Segmentation Model.pth"]
ADD ["https://insurevisstorage.blob.core.windows.net/models/Damage Type Object Detection Model.onnx","/models/Damage Type Object Detection Model.onnx"]
ADD ["https://insurevisstorage.blob.core.windows.net/models/Severity Classification Model.onnx","/models/Severity Classification Model.onnx"]
ADD ["https://insurevisstorage.blob.core.windows.net/models/Car Damage Type Segmentation Model.pth","/models/Car Damage Type Segmentation Model.pth"]


# Expose the port the app will run on
EXPOSE 5001

# The command to run the application using the Gunicorn production server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "300", "app:app"]