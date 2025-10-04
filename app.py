# app.py
import os
import io
import time
import traceback
import numpy as np
import cv2
import onnxruntime as ort
# from shapely.geometry import Polygon # Keep if needed, else remove
from PIL import Image
from typing import List, Dict, Tuple
import json

# --- Flask Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS # <-- Added CORS Import

# --- Detectron2 Imports ---
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    # Check if CUDA/MPS is available for PyTorch
    D2_DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
except ImportError as e:
    print(f"Error importing Detectron2. Please ensure it is installed along with its dependencies (like torch).")
    print(f"Details: {e}")
    exit(1) # Exit if core components are missing
except Exception as e:
    print(f"Error during Detectron2/Torch import or device check: {e}")
    exit(1)


# ===========================
# ⚙️ Load Configuration from JSON
# ===========================
CONFIG_FILE_PATH = "config.json"

def load_config(path: str) -> Dict:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded successfully from {path}")
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at {path}")
        print("Please create a config.json file based on the required structure.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Failed to decode JSON from {path}: {e}")
        print("Please check your config.json for syntax errors.")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while loading config: {e}")
        traceback.print_exc()
        exit(1)

# Load the configuration globally
CONFIG = load_config(CONFIG_FILE_PATH)

# ===========================
# ⚙️ Extract Config Values into Constants (or use CONFIG directly)
# ===========================

# --- Model Paths (Keep using environment variables or direct paths for flexibility) ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/Car Damage Estimation Models/")
# Ensure MODEL_DIR exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created model directory: {MODEL_DIR}")

PART_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Parts Segmentation Model.pth")
DAMAGE_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Damage Segmentation Model.pth")
SEVERITY_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "Severity Classification Model.onnx")
DAMAGE_TYPE_DETECT_MODEL_PATH = os.path.join(MODEL_DIR, "Damage Type Object Detection Model.onnx")


# --- Extract Model Parameters from CONFIG ---
MASKRCNN_CONFIG_FILE = CONFIG['model_params'].get('detectron2_base_config') # e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
if MASKRCNN_CONFIG_FILE is None:
    print("FATAL ERROR: 'detectron2_base_config' missing in config.json under 'model_params'.")
    exit(1)

PART_SEG_CONF_THRES = CONFIG['model_params'].get('part_seg_conf_thres', 0.5)
DAMAGE_SEG_CONF_THRES = CONFIG['model_params'].get('damage_seg_conf_thres', 0.5)
DAMAGE_DETECTOR_INPUT_SIZE = tuple(CONFIG['model_params'].get('damage_detector_input_size', [640, 640])) # Convert list to tuple
DAMAGE_DETECTOR_CONF_THRESHOLD = CONFIG['model_params'].get('damage_detector_conf_threshold', 0.5)
DAMAGE_DETECTOR_IOU_THRESHOLD = CONFIG['model_params'].get('damage_detector_iou_threshold', 0.4)
SEVERITY_CLASSIFIER_INPUT_SIZE = tuple(CONFIG['model_params'].get('severity_classifier_input_size', [224, 224]))

# --- Extract Class Names from CONFIG ---
car_part_classes = CONFIG['class_names'].get('car_parts', [])
damage_segmentation_class_names = CONFIG['class_names'].get('damage_segmentation', [])
severity_names = CONFIG['class_names'].get('severity', ["Low", "Medium", "High"])
damage_type_names = CONFIG['class_names'].get('damage_types', ["Dent", "Scratch", "Crack", "Broken"])

if not car_part_classes or not damage_segmentation_class_names or not damage_type_names:
     print("FATAL ERROR: Class names lists ('car_parts', 'damage_segmentation', 'damage_types') are missing or empty in config.json.")
     exit(1)


# --- Extract Cost Tables from CONFIG ---
part_base_costs = CONFIG['costs'].get('part_base', {})
damage_multipliers = CONFIG['costs'].get('damage_multipliers', {})
if not part_base_costs or not damage_multipliers:
     print("Warning: Cost tables ('part_base', 'damage_multipliers') are missing or empty in config.json. Cost estimation may be inaccurate.")


# --- Extract Processing Parameters from CONFIG ---
COST_ESTIMATION_IOU_THRESHOLD = CONFIG['processing_params'].get('cost_estimation_iou_threshold', 0.3)
DAMAGE_CLASS_LABEL_IN_SEGMENTER = CONFIG['processing_params'].get('damage_class_label_in_segmenter', 'Damage')

# --- Derive Damage Class Index (Important!) ---
try:
    DAMAGE_CLASS_INDEX_IN_SEGMENTER = damage_segmentation_class_names.index(DAMAGE_CLASS_LABEL_IN_SEGMENTER)
    print(f"Derived damage class index for label '{DAMAGE_CLASS_LABEL_IN_SEGMENTER}' is: {DAMAGE_CLASS_INDEX_IN_SEGMENTER}")
except ValueError:
    print(f"FATAL ERROR: Damage label '{DAMAGE_CLASS_LABEL_IN_SEGMENTER}' defined in config "
          f"not found in damage_segmentation class list: {damage_segmentation_class_names}")
    exit(1)


# ===========================
# 🧠 Model Loading (Uses constants derived from config)
# ===========================

# --- Detectron2 Model Loader Helper ---
def load_detectron2_model(config_path, weight_path, class_names_for_maskrcnn, conf_thres):
    # Check if weights file exists BEFORE attempting to load
    if not os.path.exists(weight_path):
         print(f"FATAL ERROR: Detectron2 model weights not found at: {weight_path}")
         # This is a fatal error now that download is removed.
         raise FileNotFoundError(f"Detectron2 model weights not found: {weight_path}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    try:
        cfg = get_cfg()
        # Use a local config file path or a model_zoo path
        if os.path.exists(config_path):
             cfg.merge_from_file(config_path)
             print(f"Using local Detectron2 config file: {config_path}")
        else:
             # If config_path is like "COCO-InstanceSegmentation/...", use model_zoo
             try:
                 cfg.merge_from_file(model_zoo.get_config_file(config_path))
                 print(f"Using Detectron2 model zoo config: {config_path}")
             except Exception as model_zoo_error:
                 print(f"Error accessing Detectron2 model zoo config '{config_path}': {model_zoo_error}")
                 print("Please ensure 'detectron2_base_config' in config.json is a valid model_zoo path or a local config file path.")
                 raise # Re-raise the specific model_zoo error

        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names_for_maskrcnn) # Uses loaded class names list
        cfg.MODEL.DEVICE = D2_DEVICE # Use the determined device
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loading on: {cfg.MODEL.DEVICE} with {len(class_names_for_maskrcnn)} classes.")
        predictor = DefaultPredictor(cfg)
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loaded successfully.")
        return predictor
    except FileNotFoundError: # Catch the specific FileNotFoundError raised above
        raise # Re-raise it to be caught by the main loading block
    except Exception as e:
        print(f"Error loading Detectron2 model from {weight_path}: {e}")
        traceback.print_exc()
        raise # Re-raise the exception

# --- Load All Models ---
print("Loading models...")
start_time = time.time()
try:
    # Note: File existence checks are now primarily within load_detectron2_model
    # and explicit checks before ONNX sessions.

    part_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE, # From config
        weight_path=PART_SEG_MODEL_PATH,
        class_names_for_maskrcnn=car_part_classes, # From config
        conf_thres=PART_SEG_CONF_THRES # From config
    )

    damage_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE, # From config
        weight_path=DAMAGE_SEG_MODEL_PATH,
        class_names_for_maskrcnn=damage_segmentation_class_names, # From config
        conf_thres=DAMAGE_SEG_CONF_THRES # From config
    )

    # Explicit check for ONNX models
    if not os.path.exists(SEVERITY_CLASS_MODEL_PATH):
         print(f"FATAL ERROR: Severity Classifier model not found at: {SEVERITY_CLASS_MODEL_PATH}")
         raise FileNotFoundError(f"Severity Classifier model not found: {SEVERITY_CLASS_MODEL_PATH}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    # Set ONNX Runtime provider (CPU or CUDA/MPS)
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # Prefer CUDA if available
    elif torch.backends.mps.is_available(): # Apple Silicon MPS
         providers = ['MPSExecutionProvider', 'CPUExecutionProvider'] # Prefer MPS if available (less common)
         print("Attempting to use MPSExecutionProvider for ONNX Runtime.")

    classifier_session = ort.InferenceSession(SEVERITY_CLASS_MODEL_PATH, providers=providers)
    print(f"ONNX Severity Classifier '{os.path.basename(SEVERITY_CLASS_MODEL_PATH)}' loaded with providers: {classifier_session.get_providers()}.")

    if not os.path.exists(DAMAGE_TYPE_DETECT_MODEL_PATH):
         print(f"FATAL ERROR: Damage Type Detector model not found at: {DAMAGE_TYPE_DETECT_MODEL_PATH}")
         raise FileNotFoundError(f"Damage Type Detector model not found: {DAMAGE_TYPE_DETECT_MODEL_PATH}. Ensure the model is placed in the '{MODEL_DIR}' directory.")
    detector_session = ort.InferenceSession(DAMAGE_TYPE_DETECT_MODEL_PATH, providers=providers) # Use same providers
    print(f"ONNX Damage Type Detector '{os.path.basename(DAMAGE_TYPE_DETECT_MODEL_PATH)}' loaded with providers: {detector_session.get_providers()}.")

    print(f"All models loaded successfully in {time.time() - start_time:.2f} seconds.")

# Specific check for FileNotFoundError raised by loading helpers
except FileNotFoundError as fnf_error:
    print(f"FATAL ERROR: Model file not found: {fnf_error}")
    exit(1)
except Exception as load_error:
    print(f"FATAL ERROR during model loading: {load_error}")
    traceback.print_exc()
    exit(1)

# ===========================
# 🛠️ Helper Functions (Now use global constants derived from JSON)
# ===========================

# --- NMS for ONNX Damage Detector ---
def non_max_suppression_for_damage_detector(boxes, scores, iou_threshold):
    if boxes.shape[0] == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-5)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- Post-processing for ONNX Damage Detector ---
def postprocess_onnx_damage_detections(
    outputs, original_width, original_height,
    input_width, input_height, conf_threshold,
    iou_threshold, class_names_list
):
    if not outputs or len(outputs) == 0 or outputs[0].ndim < 2:
        # print("Warning: Damage detector returned unexpected output format or is empty.") # Too verbose
        return []

    raw_output_tensor = outputs[0]

    # Try to guess the output format (common YOLO formats)
    # Format 1: (1, num_detections, 4 + num_classes)
    # Format 2: (1, 4 + num_classes, num_detections) - Transposed
    # Format 3: (num_detections, 4 + num_classes) - Single batch inference

    num_classes = len(class_names_list)
    expected_cols_or_rows = 4 + num_classes

    if raw_output_tensor.ndim == 3 and raw_output_tensor.shape[0] == 1:
        if raw_output_tensor.shape[2] == expected_cols_or_rows:
            raw_detections = raw_output_tensor[0] # Shape (num_detections, 4 + num_classes)
            # print("Detected YOLO format: (1, num_detections, 4 + num_classes)")
        # FIX: Changed expected_cols_or_classes to expected_cols_or_rows
        elif raw_output_tensor.shape[1] == expected_cols_or_rows:
            raw_detections = raw_output_tensor[0].T # Shape (num_detections, 4 + num_classes)
            # print("Detected YOLO format: (1, 4 + num_classes, num_detections), transposed.")
        else:
             print(f"Error: Damage detector unexpected output shape {raw_output_tensor.shape}. Expected {expected_cols_or_rows} in dim 1 or 2.")
             return []
    elif raw_output_tensor.ndim == 2 and raw_output_tensor.shape[1] == expected_cols_or_rows:
         raw_detections = raw_output_tensor # Shape (num_detections, 4 + num_classes)
         # print("Detected YOLO format: (num_detections, 4 + num_classes)")
    else:
        print(f"Unable to handle unexpected damage detector output shape {raw_output_tensor.shape}.")
        return []

    detections_processed = []
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    # Processing based on (num_detections, 4 + num_classes) format
    for row in raw_detections:
        if len(row) < 4 + num_classes:
            print(f"Warning: Skipping row with unexpected length {len(row)} in damage detector output.")
            continue

        box_coords, class_probs = row[:4], row[4:]

        if len(class_probs) != num_classes:
             print(f"Warning: Skipping row with incorrect number of class probabilities ({len(class_probs)} vs {num_classes}).")
             continue

        max_score_index = np.argmax(class_probs)
        max_score = class_probs[max_score_index]

        if max_score >= conf_threshold:
            class_id = max_score_index
            cx, cy, w, h = box_coords

            # Ensure bounding box dimensions are positive
            if w <= 0 or h <= 0:
                 # print(f"Warning: Skipping detection with non-positive width or height: w={w}, h={h}")
                 continue

            # Convert center-wh to min-max and scale to original image size
            x_min_inp = cx - w / 2
            y_min_inp = cy - h / 2
            x_max_inp = cx + w / 2
            y_max_inp = cy + h / 2

            # Clamp coordinates to image boundaries
            orig_x_min = max(0.0, x_min_inp * scale_x)
            orig_y_min = max(0.0, y_min_inp * scale_y)
            orig_x_max = min(original_width - 1.0, x_max_inp * scale_x)
            orig_y_max = min(original_height - 1.0, y_max_inp * scale_y)

            # Ensure calculated original size is valid
            if orig_x_max <= orig_x_min or orig_y_max <= orig_y_min:
                 # print(f"Warning: Skipping detection with invalid original coordinates: x_min={orig_x_min}, y_min={orig_y_min}, x_max={orig_x_max}, y_max={orig_y_max}")
                 continue

            detections_processed.append({
                "box": [orig_x_min, orig_y_min, orig_x_max, orig_y_max],
                "score": float(max_score),
                "class_id": int(class_id)
            })

    if not detections_processed:
        # print("Info: No detections above confidence threshold.") # Too verbose
        return []

    # Prepare for NMS
    boxes_np = np.array([d["box"] for d in detections_processed]).astype(np.float32)
    scores_np = np.array([d["score"] for d in detections_processed]).astype(np.float32)
    class_ids_np = np.array([d["class_id"] for d in detections_processed]).astype(np.int64)

    final_detections = []
    unique_classes = np.unique(class_ids_np)

    # Apply NMS per class (standard practice for object detection)
    for cls in unique_classes:
        cls_indices = np.where(class_ids_np == cls)[0]
        cls_boxes = boxes_np[cls_indices]
        cls_scores = scores_np[cls_indices]

        if cls_boxes.shape[0] > 0:
             # Apply NMS to detections of this class
             keep_indices_for_cls = non_max_suppression_for_damage_detector(
                 cls_boxes, cls_scores, iou_threshold
             )

             # Map the kept indices back to the original detections_processed list
             original_indices_kept = cls_indices[keep_indices_for_cls]

             for original_idx in original_indices_kept:
                 det = detections_processed[original_idx]
                 class_id_val = det['class_id']
                 try:
                     class_name = class_names_list[class_id_val]
                 except IndexError:
                     print(f"Warning: Damage detector class ID {class_id_val} out of bounds for {len(class_names_list)} classes. Using 'Unknown'.")
                     class_name = "Unknown" # Fallback name

                 final_detections.append({
                     "box": [int(coord) for coord in det['box']],
                     "class_name": class_name,
                     "confidence": float(det['score']),
                     "class_id": int(class_id_val)
                 })

    return final_detections


# --- Run ONNX Damage Type Detector ---
def run_onnx_damage_type_detector(image_numpy_bgr, detector_session):
    if image_numpy_bgr is None or image_numpy_bgr.size == 0:
        print("Warning: Empty image passed to run_onnx_damage_type_detector.")
        return [] # Return empty list for no detections

    original_height, original_width = image_numpy_bgr.shape[:2]
    try:
        # Convert BGR to RGB for PIL, resize, normalize
        # Ensure image_numpy_bgr is C_CONTIGUOUS
        if not image_numpy_bgr.flags['C_CONTIGUOUS']:
            image_numpy_bgr = np.ascontiguousarray(image_numpy_bgr)

        image_rgb = cv2.cvtColor(image_numpy_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        # Use global constant DAMAGE_DETECTOR_INPUT_SIZE
        img_resized_pil = img_pil.resize(DAMAGE_DETECTOR_INPUT_SIZE, Image.LANCZOS)
        img_array = np.array(img_resized_pil, dtype=np.float32) / 255.0
        img_transposed = np.transpose(img_array, (2, 0, 1)) # HWC to CHW
        input_tensor = np.expand_dims(img_transposed, axis=0) # Add batch dim

        input_name = detector_session.get_inputs()[0].name
        output_names = [out.name for out in detector_session.get_outputs()]

        outputs_onnx = detector_session.run(output_names, {input_name: input_tensor})

        # Use global constants for postprocessing parameters and class names
        detections = postprocess_onnx_damage_detections(
            outputs_onnx,
            original_width, original_height,
            DAMAGE_DETECTOR_INPUT_SIZE[0], DAMAGE_DETECTOR_INPUT_SIZE[1],
            DAMAGE_DETECTOR_CONF_THRESHOLD, # Global constant
            DAMAGE_DETECTOR_IOU_THRESHOLD,  # Global constant
            damage_type_names            # Global constant
        )

        # Return full detection dicts (with class_name and confidence)
        return detections

    except Exception as e:
        print(f"Error during ONNX Damage Type Detector inference: {e}")
        traceback.print_exc()
        return [] # Return empty list on error


# --- IoU Calculation ---
def compute_iou(mask1, mask2):
    # Ensure masks are boolean and have the same shape
    if mask1.shape != mask2.shape:
        print(f"Warning: IoU calculation received masks of different shapes: {mask1.shape} vs {mask2.shape}. Returning 0.")
        return 0.0

    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()

    # Avoid division by zero
    return intersection / union if union != 0 else 0.0 # Return float


# --- Part-Level Damage Deduplication ---
def deduplicate_damages_per_part(damages):
    """
    For each damaged part, keep only the damage detection with the highest confidence.
    This eliminates overlapping damages on the same part and improves result quality.
    """
    if not damages:
        return damages

    # Group damages by part
    damages_by_part = {}

    for damage in damages:
        part = damage.get("damaged_part", "Unknown")

        if part not in damages_by_part:
            damages_by_part[part] = []

        damages_by_part[part].append(damage)

    # Keep only highest confidence damage per part
    filtered_damages = []

    for part, part_damages in damages_by_part.items():
        if len(part_damages) == 1:
            # Only one damage for this part, keep it
            filtered_damages.append(part_damages[0])
        else:
            # Multiple damages for this part, find highest confidence
            valid_damages = []
            null_confidence_damages = []

            for damage in part_damages:
                confidence = damage.get("confidence")
                if confidence is not None:
                    valid_damages.append(damage)
                else:
                    null_confidence_damages.append(damage)

            if valid_damages:
                # Find damage with highest confidence
                best_damage = max(valid_damages, key=lambda d: d.get("confidence", 0))
                filtered_damages.append(best_damage)

                # Log what was filtered out
                filtered_out = [d for d in valid_damages if d != best_damage]
                for filtered_damage in filtered_out:
                    print(f"   🔄 Filtered overlapping damage: {filtered_damage.get('damage_type')} "
                          f"on {part} (confidence: {filtered_damage.get('confidence'):.3f}) - "
                          f"kept better: {best_damage.get('damage_type')} ({best_damage.get('confidence'):.3f})")

                # Also log null confidence damages that were filtered
                for null_damage in null_confidence_damages:
                    print(f"   🔄 Filtered null confidence damage: {null_damage.get('damage_type')} on {part}")

            elif null_confidence_damages:
                # Only null confidence damages, keep the first one
                filtered_damages.append(null_confidence_damages[0])

                for filtered_damage in null_confidence_damages[1:]:
                    print(f"   🔄 Filtered duplicate null damage: {filtered_damage.get('damage_type')} on {part}")

    return filtered_damages


# --- Image Preprocessing for Large Images ---
def resize_large_image(image_bgr, max_dimension=4000):
    """
    Resize image if it exceeds max_dimension while maintaining aspect ratio.
    This prevents timeouts and memory issues with very large images.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    height, width = image_bgr.shape[:2]
    max_current = max(height, width)

    if max_current <= max_dimension:
        print(f"Image size {width}x{height} is within limits, no resizing needed")
        return image_bgr

    # Calculate new dimensions maintaining aspect ratio
    scale_factor = max_dimension / max_current
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    print(f"Resizing large image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.3f})")

    # Use INTER_AREA for downscaling (better quality)
    resized_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


# --- Logical Consistency Filter ---
def apply_logical_consistency_filter(damages):
    """
    Apply logical consistency rules to filter out impossible damage combinations
    """
    if not damages:
        return damages

    # Define logical rules
    glass_parts = {"Windshield", "Front-window", "Back-window", "Back-windshield", "Headlight", "Tail-light"}
    plastic_metal_parts = {"Front-bumper", "Back-bumper", "Hood", "Trunk", "Fender", "Quarter-panel", "Rocker-panel", "Front-door", "Back-door"}
    lamp_parts = {"Headlight", "Tail-light"}

    filtered_damages = []

    for damage in damages:
        damage_type = damage.get("damage_type", "Unknown")
        part = damage.get("damaged_part", "unknown")
        confidence = damage.get("confidence")

        # Check for logical consistency
        is_valid, reason = is_logically_consistent(damage_type, part, confidence, glass_parts, plastic_metal_parts, lamp_parts)

        if is_valid:
            filtered_damages.append(damage)
        else:
            print(f"   🚫 Filtered illogical detection: {damage_type} on {part} - {reason}")

    return filtered_damages

def is_logically_consistent(damage_type, part, confidence, glass_parts, plastic_metal_parts, lamp_parts):
    """Check if damage type and part combination is logically consistent"""

    # Rule 1: Shattered Glass can only occur on glass parts
    if damage_type == "Shattered Glass" and part not in glass_parts:
        return False, f"Shattered glass cannot occur on {part} (not a glass part)"

    # Rule 2: Broken Lamp should primarily occur on lamp parts or bumpers (headlight/taillight integration)
    if damage_type == "Broken Lamp":
        if part == "Trunk":
            return False, "Broken lamp on trunk is highly unusual"
        if part not in lamp_parts and part not in {"Front-bumper", "Back-bumper"}:
            return False, f"Broken lamp on {part} is unusual"

    # Rule 3: Filter out Unknown damages with null confidence (high uncertainty)
    if damage_type == "Unknown" and confidence is None:
        return False, "Unknown damage type with null confidence indicates high model uncertainty"

    # Rule 4: Filter out very low confidence detections
    if confidence is not None and confidence < 0.45:
        return False, f"Very low confidence ({confidence:.3f}) indicates unreliable detection"

    # Rule 5: Damage types that don't make sense on certain parts
    if damage_type == "Flat Tire" and "wheel" not in part.lower():
        return False, f"Flat tire can only occur on wheel parts, not {part}"

    return True, "Logically consistent"

def validate_severity_consistency(damages, severity):
    """Validate that severity matches damage count and types"""

    damage_count = len(damages)

    # Rule: If severity is moderate/severe but no damages, flag as inconsistent
    if severity in ["moderate", "severe"] and damage_count == 0:
        print(f"   ⚠️  Severity-damage mismatch: '{severity}' severity with {damage_count} damages")
        return "minor"  # Downgrade severity if no damages

    # Rule: If many high-confidence damages but low severity, flag
    high_conf_damages = sum(1 for d in damages if d.get("confidence", 0) > 0.7)
    if high_conf_damages >= 2 and severity == "minor":
        print(f"   ⚠️  Severity may be underestimated: {high_conf_damages} high-confidence damages but '{severity}' severity")

    return severity


# --- Run ONNX Severity Classifier ---
def run_yolo_classifier(image, classifier_session):
    if image is None or image.size == 0:
        print("Warning: Empty image passed to run_yolo_classifier.")
        return 0 # Default to first class (e.g., Low)

    try:
        # Ensure image is C_CONTIGUOUS
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Use global constant SEVERITY_CLASSIFIER_INPUT_SIZE
        img_resized = cv2.resize(image, SEVERITY_CLASSIFIER_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Classifier expects RGB
        img_normalized = img_rgb / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1).astype(np.float32)[None, :] # CHW, Add batch dim

        input_name = classifier_session.get_inputs()[0].name
        outputs = classifier_session.run(None, {input_name: img_transposed})

        # Assuming the output is a single tensor of shape (1, num_classes)
        if outputs and len(outputs) > 0 and outputs[0].ndim == 2 and outputs[0].shape[0] == 1:
            # Apply softmax if needed (usually included in the model, but argmax on logits is common)
            probabilities = outputs[0][0]
            if np.any(probabilities < 0) or np.max(np.abs(probabilities)) > 100:
                 e_x = np.exp(probabilities - np.max(probabilities))
                 probabilities = e_x / e_x.sum()

            return int(np.argmax(probabilities)) # Get index of max probability
        else:
             print(f"Warning: Unexpected output shape from classifier: {outputs[0].shape if outputs else 'None'}")
             return 0 # Default to first class

    except Exception as e:
        print(f"Error during ONNX Severity Classifier inference: {e}")
        traceback.print_exc()
        return 0 # Default to first class on error


# --- Run Detectron2 Mask R-CNN ---
def run_mask_rcnn(image, predictor):
    if image is None or image.size == 0:
        print("Warning: Empty image passed to run_mask_rcnn.")
        return np.array([]), np.array([])

    try:
        # Ensure image is C_CONTIGUOUS for Detectron2
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu") # Always move results to CPU
        if not instances.has("pred_masks") or not instances.has("pred_classes"):
             # print("Warning: Detectron2 output missing 'pred_masks' or 'pred_classes'.") # Too verbose
             return np.array([]), np.array([])
        masks = instances.pred_masks.numpy() # Shape (N, H, W) boolean
        classes = instances.pred_classes.numpy() # Shape (N,) int

        return masks, classes
    except Exception as e:
        print(f"Error during Detectron2 inference: {e}")
        traceback.print_exc()
        return np.array([]), np.array([])


# --- Estimate Repair Cost ---

# ===========================
#  Flask Application Setup
# ===========================
app = Flask(__name__)
CORS(app) # Apply CORS to your app

@app.route('/')
def home():
    # Add simple check for models existing on startup
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH, DAMAGE_TYPE_DETECT_MODEL_PATH]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    status_message = "Car Damage Estimation API is running."
    overall_status = "OK"
    if missing_models:
        status_message += f" WARNING: The following model files are missing: {', '.join(missing_models)}. Prediction requests may fail."
        overall_status = "WARNING: Models Missing"

    return jsonify({"message": status_message, "status": overall_status}), 200 if overall_status == "OK" else 500


@app.route('/predict', methods=['POST'])
def predict_damage_cost():
    print("\nReceived request on /predict")
    start_req_time = time.time()
    if 'part_predictor' not in globals() or 'damage_predictor' not in globals() or \
       'classifier_session' not in globals() or 'detector_session' not in globals():
        print("Error: Models are not loaded. Check server startup logs.")
        return jsonify({"error": "Server internal error: Models not loaded."}), 500
    if 'image_file' not in request.files:
        print("Error: No 'image_file' part in the request.")
        return jsonify({"error": "Missing 'image_file' in request form data"}), 400
    file = request.files['image_file']
    if file.filename == '':
        print("Error: No selected file.")
        return jsonify({"error": "No selected file"}), 400
    try:
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
            if len(image_bytes) == 0:
                raise ValueError("Uploaded file is empty.")
            else:
                raise ValueError("Could not decode image. Invalid format?")
        print(f"Image decoded successfully: shape={image_bgr.shape}, dtype={image_bgr.dtype}")

        # Resize large images to prevent timeouts and memory issues
        original_shape = image_bgr.shape
        image_bgr = resize_large_image(image_bgr, max_dimension=4000)
        if image_bgr.shape != original_shape:
            print(f"Image resized for processing: {original_shape} -> {image_bgr.shape}")
    except Exception as e:
        print(f"Error reading/decoding image: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to read or decode image: {e}"}), 400
    try:
        # Run part segmentation
        print("Running Part Segmentation...")
        part_masks, part_class_idxs = run_mask_rcnn(image_bgr, part_predictor)
        print(f"Part Segmentation found {len(part_masks)} masks.")
        # Run damage segmentation
        print("Running Damage Segmentation...")
        all_damage_masks, all_damage_classes = run_mask_rcnn(image_bgr, damage_predictor)
        damage_masks = np.array([])
        if all_damage_masks.size > 0 and all_damage_classes.size > 0 and len(all_damage_masks) == len(all_damage_classes):
            if not all_damage_masks.flags['C_CONTIGUOUS']:
                all_damage_masks = np.ascontiguousarray(all_damage_masks)
            damage_masks = all_damage_masks[all_damage_classes == DAMAGE_CLASS_INDEX_IN_SEGMENTER]
            print(f"Damage Segmentation found {len(damage_masks)} damage masks.")
        else:
            print("No valid damage masks found.")
        print(f"Part masks count: {len(part_masks)}")
        print(f"Damage masks count: {len(damage_masks)}")
        # Map part class indices to labels
        part_labels = []
        if part_class_idxs.size > 0:
            for idx in part_class_idxs:
                if 0 <= idx < len(car_part_classes):
                    part_labels.append(car_part_classes[idx])
                else:
                    part_labels.append(f"Unknown_Part_Idx_{idx}")
        # Prepare damage results
        damages = []
        overlap_count = 0
        if damage_masks.size > 0 and part_masks.size > 0:
            for i, dmg_mask in enumerate(damage_masks):
                for j, part_mask in enumerate(part_masks):
                    iou = compute_iou(dmg_mask.astype(bool), part_mask.astype(bool))
                    if iou > COST_ESTIMATION_IOU_THRESHOLD:
                        overlap_count += 1
                        part_name = part_labels[j] if j < len(part_labels) else f"Part_{j}"
                        overlap_mask_bool = np.logical_and(dmg_mask.astype(bool), part_mask.astype(bool))
                        rows, cols = np.where(overlap_mask_bool)
                        if rows.size == 0 or cols.size == 0:
                            continue
                        y_min, y_max = int(np.min(rows)), int(np.max(rows))
                        x_min, x_max = int(np.min(cols)), int(np.max(cols))
                        padding = 5
                        img_h, img_w = image_bgr.shape[:2]
                        crop_y_min = max(0, y_min - padding)
                        crop_y_max = min(img_h, y_max + 1 + padding)
                        crop_x_min = max(0, x_min - padding)
                        crop_x_max = min(img_w, x_max + 1 + padding)
                        if crop_y_max <= crop_y_min or crop_x_max <= crop_x_min:
                            continue
                        cropped_image_bgr = image_bgr[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
                        detected_damage_detections = run_onnx_damage_type_detector(cropped_image_bgr, detector_session)
                        if not detected_damage_detections:
                            detected_damage_detections = [{"class_name": "Unknown", "confidence": None}]
                        for det in detected_damage_detections:
                            damages.append({
                                "damage_type": det.get("class_name", "Unknown"),
                                "confidence": det.get("confidence", None),
                                "damaged_part": part_name,
                                "bounding_box": [crop_x_min, crop_y_min, crop_x_max, crop_y_max]
                            })
        print(f"Total overlaps found: {overlap_count}")

        # Apply part-level deduplication to remove overlapping damages
        print("Applying part-level damage deduplication...")
        original_damage_count = len(damages)
        damages = deduplicate_damages_per_part(damages)
        duplicates_removed = original_damage_count - len(damages)
        print(f"Deduplication complete: {original_damage_count} → {len(damages)} damages ({duplicates_removed} duplicates removed)")

        # Apply logical consistency filter to remove impossible combinations
        print("Applying logical consistency filter...")
        pre_filter_count = len(damages)
        damages = apply_logical_consistency_filter(damages)
        logical_filtered = pre_filter_count - len(damages)
        print(f"Logical filtering complete: {pre_filter_count} → {len(damages)} damages ({logical_filtered} illogical detections removed)")

        # Determine overall severity
        print("Determining overall severity...")
        t_severity = time.time()
        overall_severity_index = run_yolo_classifier(image_bgr, classifier_session)
        overall_severity_name = "Unknown"
        if 0 <= overall_severity_index < len(severity_names):
            overall_severity_name = severity_names[overall_severity_index]
        else:
            print(f"Warning: Overall severity index {overall_severity_index} out of bounds for {len(severity_names)} severity names. Using 'Unknown'.")

        # Validate severity consistency
        overall_severity_name = validate_severity_consistency(damages, overall_severity_name)

        final_result = {
            "overall_severity": overall_severity_name,
            "damages": damages if damages else []
        }
        print(f"Overall severity determined in {time.time()-t_severity:.2f}s: {overall_severity_name}")
        total_processing_time = time.time() - start_req_time
        print(f"Request processed successfully in {total_processing_time:.2f} seconds.")
        print("--- Final Result about to be JSONified ---")
        print(json.dumps(final_result, indent=2))
        print("------------------------------------------")
        return jsonify(final_result), 200
    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during processing.", "details": str(e)}), 500


# ===========================
# Main Driver
# ===========================

if __name__ == '__main__':
    print("Starting Car Damage Estimation API...")

    # Add a quick check on startup to remind the user if models are missing
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH, DAMAGE_TYPE_DETECT_MODEL_PATH]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        print("\n" + "="*50)
        print("!!! WARNING: Missing Model Files !!!")
        print("The application is configured to *load* models from specific paths,")
        print("but the following expected model files were NOT FOUND:")
        for mf in missing_models:
            print(f"- {mf}")
        print(f"Please ensure the models are placed in the '{MODEL_DIR}' directory.")
        print("Prediction requests will likely fail until these files are present.")
        print("="*50 + "\n")

    # Determine the port Flask will run on
    # Get port from environment variable PORT, default to 5001
    flask_port = int(os.getenv("PORT", 5001))
    flask_host = "0.0.0.0" # Bind to all interfaces

    print(f"Starting Flask server on {flask_host}:{flask_port}")
    print(f"You can access the API locally at http://localhost:{flask_port}")

    # Run the Flask app directly. This is a blocking call.
    # Use debug=False for production environments.
    app.run(host=flask_host, port=flask_port, debug=False)

    print("Server has been shut down.")