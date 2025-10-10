# app.py
import os
import io
import time
import traceback
import numpy as np
import cv2
import onnxruntime as ort
# from shapely.geometry import Polygon # Keep if needed, else remove
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
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/")
# Ensure MODEL_DIR exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created model directory: {MODEL_DIR}")

PART_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Parts Segmentation Model.pth")
DAMAGE_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Damage Type Segmentation Model.pth")
SEVERITY_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "Severity Classification Model.onnx")


# --- Extract Model Parameters from CONFIG ---
MASKRCNN_CONFIG_FILE = CONFIG['model_params'].get('detectron2_base_config') # e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
if MASKRCNN_CONFIG_FILE is None:
    print("FATAL ERROR: 'detectron2_base_config' missing in config.json under 'model_params'.")
    exit(1)

PART_SEG_CONF_THRES = CONFIG['model_params'].get('part_seg_conf_thres', 0.5)
DAMAGE_SEG_CONF_THRES = CONFIG['model_params'].get('damage_seg_conf_thres', 0.5)
SEVERITY_CLASSIFIER_INPUT_SIZE = tuple(CONFIG['model_params'].get('severity_classifier_input_size', [224, 224]))

# --- Extract Class Names from CONFIG ---
car_part_classes = CONFIG['class_names'].get('car_parts', [])
damage_segmentation_class_names = CONFIG['class_names'].get('damage_segmentation', [])
severity_names = CONFIG['class_names'].get('severity', ["Low", "Medium", "High"])

if not car_part_classes or not damage_segmentation_class_names:
     print("FATAL ERROR: Class names lists ('car_parts', 'damage_segmentation') are missing or empty in config.json.")
     exit(1)


# --- Extract Cost Tables from CONFIG ---
part_base_costs = CONFIG['costs'].get('part_base', {})
damage_multipliers = CONFIG['costs'].get('damage_multipliers', {})
if not part_base_costs or not damage_multipliers:
     print("Warning: Cost tables ('part_base', 'damage_multipliers') are missing or empty in config.json. Cost estimation may be inaccurate.")


# --- Extract Processing Parameters from CONFIG ---
COST_ESTIMATION_IOU_THRESHOLD = CONFIG['processing_params'].get('cost_estimation_iou_threshold', 0.3)


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
    Apply logical consistency rules to filter out impossible damage combinations.
    Rules enforce physical constraints and spatial coherence:
    - Damage types must be compatible with parts (e.g., no "Shattered Glass" on "Hood")
    - Front/rear parts shouldn't mix in the same damage set (spatial coherence)
    - Specific damage-part combinations that are physically impossible are filtered
    """
    if not damages:
        return damages

    # Define valid damage-part combinations
    PART_DAMAGE_COMPATIBILITY = {
        # Glass-related parts can only have glass damage types
        "Back Window": ["Shattered Glass", "Crack"],
        "Front Window": ["Shattered Glass", "Crack"],
        "Back Windshield": ["Shattered Glass", "Crack"],
        "Windshield": ["Shattered Glass", "Crack"],
        
        # Lamps can have specific damage types
        "Headlight": ["Broken Lamp", "Crack", "Shattered Glass"],
        "Tail Light": ["Broken Lamp", "Crack", "Shattered Glass"],
        
        # Wheels have specific damage patterns
        "Front Wheel": ["Flat Tire", "Scratch / Paint Wear", "Dent"],
        "Back Wheel": ["Flat Tire", "Scratch / Paint Wear", "Dent"],
        
        # Body panels - no glass or lamp damage
        "Hood": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Trunk": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Roof": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Fender": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Quarter Panel": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Rocker Panel": ["Dent", "Scratch / Paint Wear", "Crack"],
        
        # Doors can have various damage
        "Front Door": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Back Door": ["Dent", "Scratch / Paint Wear", "Crack"],
        
        # Bumpers and grille
        "Front Bumper": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Back Bumper": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Grille": ["Crack", "Dent", "Scratch / Paint Wear"],
        
        # Mirrors and license plates
        "Mirror": ["Shattered Glass", "Crack", "Scratch / Paint Wear"],
        "License Plate": ["Dent", "Scratch / Paint Wear"],
    }
    
    # Define front/rear part categories for spatial coherence
    FRONT_PARTS = {
        "Hood", "Front Bumper", "Front Door", "Front Wheel", 
        "Front Window", "Grille", "Headlight", "Windshield", "Fender"
    }
    
    REAR_PARTS = {
        "Trunk", "Back Bumper", "Back Door", "Back Wheel",
        "Back Window", "Back Windshield", "Tail Light", "Quarter Panel"
    }
    
    filtered_damages = []
    removed_count = 0
    
    # Step 1: Filter by damage-part compatibility
    compatible_damages = []
    for damage in damages:
        damage_type = damage.get("damage_type", "Unknown")
        part = damage.get("damaged_part", "Unknown")
        confidence = damage.get("confidence", 0)
        
        # Check if this damage-part combination is valid
        if part in PART_DAMAGE_COMPATIBILITY:
            allowed_damages = PART_DAMAGE_COMPATIBILITY[part]
            if damage_type in allowed_damages:
                compatible_damages.append(damage)
            else:
                removed_count += 1
                print(f"   🚫 Filtered incompatible: {damage_type} on {part} (not allowed)")
        else:
            # Unknown part - keep it but log
            compatible_damages.append(damage)
            if part not in ["Unknown", "Background"]:
                print(f"   ⚠️  Unknown part '{part}' - keeping damage but not validated")
    
    # Step 2: Check spatial coherence (front vs rear)
    if len(compatible_damages) > 1:
        # Count front and rear parts
        front_count = sum(1 for d in compatible_damages if d.get("damaged_part") in FRONT_PARTS)
        rear_count = sum(1 for d in compatible_damages if d.get("damaged_part") in REAR_PARTS)
        
        # If we have both front and rear damages, filter out the minority
        if front_count > 0 and rear_count > 0:
            # Determine which side has more damage
            if front_count > rear_count:
                # Keep front, remove rear
                print(f"   🔍 Spatial coherence: Detected {front_count} front + {rear_count} rear parts")
                print(f"   🚫 Filtering out rear parts (minority)")
                for damage in compatible_damages:
                    part = damage.get("damaged_part")
                    if part not in REAR_PARTS:
                        filtered_damages.append(damage)
                    else:
                        removed_count += 1
                        print(f"   🚫 Filtered spatial outlier: {damage.get('damage_type')} on {part}")
            elif rear_count > front_count:
                # Keep rear, remove front
                print(f"   🔍 Spatial coherence: Detected {front_count} front + {rear_count} rear parts")
                print(f"   🚫 Filtering out front parts (minority)")
                for damage in compatible_damages:
                    part = damage.get("damaged_part")
                    if part not in FRONT_PARTS:
                        filtered_damages.append(damage)
                    else:
                        removed_count += 1
                        print(f"   🚫 Filtered spatial outlier: {damage.get('damage_type')} on {part}")
            else:
                # Equal counts - keep all (ambiguous case)
                filtered_damages = compatible_damages
                print(f"   ⚠️  Spatial ambiguity: Equal front ({front_count}) and rear ({rear_count}) parts - keeping all")
        else:
            # All same side or neutral - keep all
            filtered_damages = compatible_damages
    else:
        # 0 or 1 damage - no spatial filtering needed
        filtered_damages = compatible_damages
    
    if removed_count > 0:
        print(f"   ✅ Logical consistency: Filtered {removed_count} inconsistent detections")
    
    return filtered_damages

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
        return np.array([]), np.array([]), np.array([])

    try:
        # Ensure image is C_CONTIGUOUS for Detectron2
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu") # Always move results to CPU
        if not instances.has("pred_masks") or not instances.has("pred_classes"):
             # print("Warning: Detectron2 output missing 'pred_masks' or 'pred_classes'.") # Too verbose
             return np.array([]), np.array([]), np.array([])
        masks = instances.pred_masks.numpy() # Shape (N, H, W) boolean
        classes = instances.pred_classes.numpy() # Shape (N,) int
        scores = instances.scores.numpy() if instances.has("scores") else np.array([])

        return masks, classes, scores
    except Exception as e:
        print(f"Error during Detectron2 inference: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([])


# --- Estimate Repair Cost ---

# ===========================
#  Flask Application Setup
# ===========================
app = Flask(__name__)
CORS(app) # Apply CORS to your app

@app.route('/')
def home():
    # Add simple check for models existing on startup
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH]
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
       'classifier_session' not in globals():
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
        part_masks, part_class_idxs, part_scores = run_mask_rcnn(image_bgr, part_predictor)
        print(f"Part Segmentation found {len(part_masks)} masks.")
        # Run damage segmentation
        print("Running Damage Segmentation...")
        all_damage_masks, all_damage_class_idxs, all_damage_scores = run_mask_rcnn(image_bgr, damage_predictor)
        
        # Filter out "Background" class (index 0) - we only want actual damage types
        # Note: Since config.json no longer has "Background" at index 0, we need to subtract 1 from class indices
        if all_damage_masks.size > 0 and all_damage_class_idxs.size > 0:
            # Keep only non-background damage masks
            non_background_indices = all_damage_class_idxs != 0
            damage_masks = all_damage_masks[non_background_indices]
            damage_class_idxs = all_damage_class_idxs[non_background_indices] - 1  # Subtract 1 to match config indices
            damage_scores = all_damage_scores[non_background_indices] if all_damage_scores.size > 0 else np.array([])
            print(f"Damage Segmentation found {len(all_damage_masks)} total masks, {len(damage_masks)} actual damage masks (filtered out background).")
        else:
            damage_masks = np.array([])
            damage_class_idxs = np.array([])
            damage_scores = np.array([])
            print("No damage masks found.")
        
        print(f"Part masks count: {len(part_masks)}")
        print(f"Damage masks count: {len(damage_masks)}")
        
        # Map damage class indices to labels
        damage_type_labels = []
        if damage_class_idxs.size > 0:
            for idx in damage_class_idxs:
                if 0 <= idx < len(damage_segmentation_class_names):
                    damage_type_labels.append(damage_segmentation_class_names[idx])
                else:
                    damage_type_labels.append(f"Unknown_Damage_Idx_{idx}")
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
                # Get the damage type for this mask
                damage_type = damage_type_labels[i] if i < len(damage_type_labels) else "Unknown"
                # Get the confidence score for this damage detection
                damage_confidence = float(damage_scores[i]) if i < len(damage_scores) else None
                
                for j, part_mask in enumerate(part_masks):
                    iou = compute_iou(dmg_mask.astype(bool), part_mask.astype(bool))
                    if iou > COST_ESTIMATION_IOU_THRESHOLD:
                        overlap_count += 1
                        part_name = part_labels[j] if j < len(part_labels) else f"Part_{j}"
                        # Get the confidence score for this part detection
                        part_confidence = float(part_scores[j]) if j < len(part_scores) else None
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
                        
                        # Record the damaged part with its damage type from segmentation
                        damages.append({
                            "damage_type": damage_type,
                            "confidence": damage_confidence,
                            "damaged_part": part_name,
                            "part_confidence": part_confidence,
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
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH]
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