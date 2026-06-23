# InsureVis Model API

InsureVis Model API is a Flask-based computer vision service for vehicle damage assessment.  
It loads pre-trained segmentation/classification models, analyzes an uploaded vehicle image, and returns:

- overall damage severity (`minor`, `moderate`, `severe`)
- detected damage entries with damage type, damaged part, confidence scores, and bounding boxes

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Model Files](#model-files)
- [Configuration](#configuration)
- [Running the API Locally](#running-the-api-locally)
- [Running with Docker](#running-with-docker)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This API is designed for insurance/inspection workflows where a single car image is uploaded and analyzed using three model components:

1. **Car part segmentation** (Detectron2 Mask R-CNN)
2. **Damage type segmentation** (Detectron2 Mask R-CNN)
3. **Severity classification** (ONNX model via ONNX Runtime)

The service performs geometric overlap checks between detected part masks and damage masks to associate each detected damage with a specific vehicle part.

---

## How It Works

1. Load configuration from `config.json`
2. Load model files from `MODEL_DIR` (default: `/models/`)
3. Receive image from `POST /predict` as `multipart/form-data`
4. Decode image and optionally downscale very large images
5. Run part segmentation + damage segmentation
6. Match damages to parts using IoU threshold from config
7. Run logical post-filters and deduplication
8. Run severity classifier
9. Return structured JSON response

---

## Tech Stack

- **Python 3.9**
- **Flask + Flask-CORS**
- **Detectron2 + PyTorch**
- **ONNX Runtime**
- **OpenCV / NumPy**
- **Gunicorn** (production serving)

Dependencies are listed in `requirements.txt`.

---

## Repository Structure

```text
insurevis-model-api/
├── app.py            # Main Flask app, model loading, inference pipeline
├── config.json       # Class names, thresholds, costs, and processing parameters
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build and production run command
└── README.md
```

---

## Requirements

### System-level

For local/manual setup, ensure these packages are available (also installed in Docker):

- `build-essential`
- `git`
- `libgl1`
- `libglib2.0-0`

### Python-level

Install:

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
```

> `detectron2` is installed from source via GitHub as declared in `requirements.txt`.

---

## Model Files

At startup, the app expects these files inside `MODEL_DIR` (default `/models/`):

- `Car Parts Segmentation Model.pth`
- `Car Damage Type Segmentation Model.pth`
- `Severity Classification Model.onnx`

If required model files are missing, startup fails or health status returns warning.

---

## Configuration

`config.json` controls:

- class name mappings for parts, damage types, and severity classes
- part base costs and damage multipliers
- model confidence thresholds and input sizes
- processing thresholds (e.g., IoU threshold for part/damage matching)

Key sections:

- `class_names`
- `costs`
- `model_params`
- `processing_params`

---

## Running the API Locally

From repository root:

```bash
export MODEL_DIR=/absolute/path/to/models
export PORT=5001
python app.py
```

The app starts on `0.0.0.0:$PORT` (default `5001` when running directly with Python).

---

## Running with Docker

Build image:

```bash
docker build -t insurevis-model-api .
```

Run container:

```bash
docker run --rm -p 8080:8080 \
  -e MODEL_DIR=/models \
  -v /absolute/path/to/models:/models \
  insurevis-model-api
```

The Docker image uses Gunicorn:

```bash
gunicorn --bind :$PORT --workers 3 --threads 4 --timeout 120 --preload app:app
```

---

## Environment Variables

- `MODEL_DIR`  
  Directory containing model files. Default: `/models/`
- `PORT`  
  Server port. Dockerfile default: `8080`; direct `python app.py` default: `5001`

---

## API Reference

### `GET /`

Health/status endpoint.

**Response example**

```json
{
  "message": "Car Damage Estimation API is running.",
  "status": "OK"
}
```

Returns status warning when expected model files are missing.

### `POST /predict`

Run damage analysis on a single image.

**Request**

- Content-Type: `multipart/form-data`
- Field name: `image_file`

**cURL example**

```bash
curl -X POST http://localhost:5001/predict \
  -F "image_file=@/absolute/path/to/car-image.jpg"
```

**Success response example**

```json
{
  "overall_severity": "moderate",
  "damages": [
    {
      "damage_type": "Dent",
      "confidence": 0.91,
      "damaged_part": "Front Bumper",
      "part_confidence": 0.88,
      "bounding_box": [120, 90, 260, 220]
    }
  ]
}
```

---

## Error Handling

Common error responses from `POST /predict`:

- `400` when:
  - `image_file` is missing
  - selected file is empty
  - image decoding fails
- `500` when:
  - models were not loaded successfully
  - internal inference pipeline failure occurs

---

## Operational Notes

- A global lock is used around segmentation model inference for safer concurrent serving.
- Large images are downscaled when max dimension exceeds 4000 pixels.
- Damage-part matching is based on IoU and post-filtered for logical consistency.
- CORS is enabled globally.

---

## Troubleshooting

- **Startup fails with missing model files**  
  Verify `MODEL_DIR` and model filenames exactly match expected names.

- **Detectron2 import/build errors**  
  Ensure PyTorch and system build dependencies are installed before installing `requirements.txt`.

- **Slow inference / memory pressure**  
  Use the provided Gunicorn settings and avoid oversized model files or insufficient container memory.