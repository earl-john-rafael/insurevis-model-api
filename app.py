# app.py
import os
import io
import time
import traceback
import numpy as np
import cv2
import onnxruntime as ort
import json
import threading  # <--- NEW: For thread safety

# from shapely.geometry import Polygon # Keep if needed, else remove
from typing import List, Dict, Tuple

# --- Flask Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS

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
# ⚙️ Extract Config Values into Constants
# ===========================

# --- Model Paths ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/")
# Ensure MODEL_DIR exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created model directory: {MODEL_DIR}")

PART_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Parts Segmentation Model.pth")
DAMAGE_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "Car Damage Type Segmentation Model.pth")
SEVERITY_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "Severity Classification Model.onnx")


# --- Extract Model Parameters from CONFIG ---
MASKRCNN_CONFIG_FILE = CONFIG['model_params'].get('detectron2_base_config')
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
     print("Warning: Cost tables ('part_base', 'damage_multipliers') are missing or empty in config.json.")


# --- Extract Processing Parameters from CONFIG ---
COST_ESTIMATION_IOU_THRESHOLD = CONFIG['processing_params'].get('cost_estimation_iou_threshold', 0.3)


# ===========================
# 🧠 Model Loading
# ===========================

# --- Detectron2 Model Loader Helper ---
def load_detectron2_model(config_path, weight_path, class_names_for_maskrcnn, conf_thres):
    if not os.path.exists(weight_path):
         print(f"FATAL ERROR: Detectron2 model weights not found at: {weight_path}")
         raise FileNotFoundError(f"Detectron2 model weights not found: {weight_path}.")
    try:
        cfg = get_cfg()
        if os.path.exists(config_path):
             cfg.merge_from_file(config_path)
             print(f"Using local Detectron2 config file: {config_path}")
        else:
             try:
                 cfg.merge_from_file(model_zoo.get_config_file(config_path))
                 print(f"Using Detectron2 model zoo config: {config_path}")
             except Exception as model_zoo_error:
                 print(f"Error accessing Detectron2 model zoo config '{config_path}': {model_zoo_error}")
                 raise

        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names_for_maskrcnn)
        cfg.MODEL.DEVICE = D2_DEVICE
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loading on: {cfg.MODEL.DEVICE} with {len(class_names_for_maskrcnn)} classes.")
        predictor = DefaultPredictor(cfg)
        print(f"Detectron2 model '{os.path.basename(weight_path)}' loaded successfully.")
        return predictor
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"Error loading Detectron2 model from {weight_path}: {e}")
        traceback.print_exc()
        raise

# --- Load All Models ---
print("Loading models...")
start_time = time.time()

# NEW: Global Lock for Inference
model_lock = threading.Lock()

try:
    part_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE,
        weight_path=PART_SEG_MODEL_PATH,
        class_names_for_maskrcnn=car_part_classes,
        conf_thres=PART_SEG_CONF_THRES
    )

    damage_predictor = load_detectron2_model(
        config_path=MASKRCNN_CONFIG_FILE,
        weight_path=DAMAGE_SEG_MODEL_PATH,
        class_names_for_maskrcnn=damage_segmentation_class_names,
        conf_thres=DAMAGE_SEG_CONF_THRES
    )

    if not os.path.exists(SEVERITY_CLASS_MODEL_PATH):
         print(f"FATAL ERROR: Severity Classifier model not found at: {SEVERITY_CLASS_MODEL_PATH}")
         raise FileNotFoundError(f"Severity Classifier model not found: {SEVERITY_CLASS_MODEL_PATH}.")
    
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif torch.backends.mps.is_available():
         providers = ['MPSExecutionProvider', 'CPUExecutionProvider']
         print("Attempting to use MPSExecutionProvider for ONNX Runtime.")

    classifier_session = ort.InferenceSession(SEVERITY_CLASS_MODEL_PATH, providers=providers)
    print(f"ONNX Severity Classifier '{os.path.basename(SEVERITY_CLASS_MODEL_PATH)}' loaded with providers: {classifier_session.get_providers()}.")

    print(f"All models loaded successfully in {time.time() - start_time:.2f} seconds.")

except FileNotFoundError as fnf_error:
    print(f"FATAL ERROR: Model file not found: {fnf_error}")
    exit(1)
except Exception as load_error:
    print(f"FATAL ERROR during model loading: {load_error}")
    traceback.print_exc()
    exit(1)

# ===========================
# 🛠️ Helper Functions
# ===========================

def compute_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        print(f"Warning: IoU calculation received masks of different shapes. Returning 0.")
        return 0.0
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    return intersection / union if union != 0 else 0.0

def deduplicate_damages_per_part(damages):
    if not damages:
        return damages
    damages_by_part = {}
    for damage in damages:
        part = damage.get("damaged_part", "Unknown")
        if part not in damages_by_part:
            damages_by_part[part] = []
        damages_by_part[part].append(damage)

    filtered_damages = []
    for part, part_damages in damages_by_part.items():
        if len(part_damages) == 1:
            filtered_damages.append(part_damages[0])
        else:
            valid_damages = [d for d in part_damages if d.get("confidence") is not None]
            null_confidence_damages = [d for d in part_damages if d.get("confidence") is None]

            if valid_damages:
                best_damage = max(valid_damages, key=lambda d: d.get("confidence", 0))
                filtered_damages.append(best_damage)
            elif null_confidence_damages:
                filtered_damages.append(null_confidence_damages[0])
    return filtered_damages

def resize_large_image(image_bgr, max_dimension=4000):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    max_current = max(height, width)
    if max_current <= max_dimension:
        return image_bgr
    scale_factor = max_dimension / max_current
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    print(f"Resizing large image from {width}x{height} to {new_width}x{new_height}")
    resized_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def apply_logical_consistency_filter(damages):
    if not damages:
        return damages
    # [Truncated list for brevity, keeping your existing logic structure]
    PART_DAMAGE_COMPATIBILITY = {
        "Back Window": ["Shattered Glass", "Crack"], "Front Window": ["Shattered Glass", "Crack"],
        "Back Windshield": ["Shattered Glass", "Crack"], "Windshield": ["Shattered Glass", "Crack"],
        "Headlight": ["Broken Lamp", "Crack", "Shattered Glass"], "Tail Light": ["Broken Lamp", "Crack", "Shattered Glass"],
        "Front Wheel": ["Flat Tire", "Scratch / Paint Wear", "Dent"], "Back Wheel": ["Flat Tire", "Scratch / Paint Wear", "Dent"],
        "Hood": ["Dent", "Scratch / Paint Wear", "Crack"], "Trunk": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Roof": ["Dent", "Scratch / Paint Wear", "Crack"], "Fender": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Quarter Panel": ["Dent", "Scratch / Paint Wear", "Crack"], "Rocker Panel": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Front Door": ["Dent", "Scratch / Paint Wear", "Crack"], "Back Door": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Front Bumper": ["Dent", "Scratch / Paint Wear", "Crack"], "Back Bumper": ["Dent", "Scratch / Paint Wear", "Crack"],
        "Grille": ["Crack", "Dent", "Scratch / Paint Wear"], "Mirror": ["Shattered Glass", "Crack", "Scratch / Paint Wear"],
        "License Plate": ["Dent", "Scratch / Paint Wear"],
    }
    FRONT_PARTS = {"Hood", "Front Bumper", "Front Door", "Front Wheel", "Front Window", "Grille", "Headlight", "Windshield", "Fender"}
    REAR_PARTS = {"Trunk", "Back Bumper", "Back Door", "Back Wheel", "Back Window", "Back Windshield", "Tail Light", "Quarter Panel"}
    
    filtered_damages = []
    compatible_damages = []
    
    # 1. Compatibility
    for damage in damages:
        damage_type = damage.get("damage_type", "Unknown")
        part = damage.get("damaged_part", "Unknown")
        if part in PART_DAMAGE_COMPATIBILITY:
            if damage_type in PART_DAMAGE_COMPATIBILITY[part]:
                compatible_damages.append(damage)
        else:
            compatible_damages.append(damage)
    
    # 2. Spatial Coherence
    if len(compatible_damages) > 1:
        front_count = sum(1 for d in compatible_damages if d.get("damaged_part") in FRONT_PARTS)
        rear_count = sum(1 for d in compatible_damages if d.get("damaged_part") in REAR_PARTS)
        if front_count > 0 and rear_count > 0:
            if front_count > rear_count:
                for damage in compatible_damages:
                    if damage.get("damaged_part") not in REAR_PARTS:
                        filtered_damages.append(damage)
            elif rear_count > front_count:
                for damage in compatible_damages:
                    if damage.get("damaged_part") not in FRONT_PARTS:
                        filtered_damages.append(damage)
            else:
                filtered_damages = compatible_damages
        else:
            filtered_damages = compatible_damages
    else:
        filtered_damages = compatible_damages
        
    return filtered_damages

def validate_severity_consistency(damages, severity):
    damage_count = len(damages)
    if severity in ["moderate", "severe"] and damage_count == 0:
        return "minor"
    high_conf_damages = sum(1 for d in damages if d.get("confidence", 0) > 0.7)
    if high_conf_damages >= 2 and severity == "minor":
        pass # Could log warning here
    return severity

def run_yolo_classifier(image, classifier_session):
    if image is None or image.size == 0: return 0
    try:
        if not image.flags['C_CONTIGUOUS']: image = np.ascontiguousarray(image)
        img_resized = cv2.resize(image, SEVERITY_CLASSIFIER_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1).astype(np.float32)[None, :]
        input_name = classifier_session.get_inputs()[0].name
        outputs = classifier_session.run(None, {input_name: img_transposed})
        if outputs and len(outputs) > 0 and outputs[0].ndim == 2:
            probabilities = outputs[0][0]
            if np.any(probabilities < 0) or np.max(np.abs(probabilities)) > 100:
                 e_x = np.exp(probabilities - np.max(probabilities))
                 probabilities = e_x / e_x.sum()
            return int(np.argmax(probabilities))
        return 0
    except Exception as e:
        print(f"Error during ONNX Severity Classifier: {e}")
        return 0

def run_mask_rcnn(image, predictor):
    if image is None or image.size == 0: return np.array([]), np.array([]), np.array([])
    try:
        if not image.flags['C_CONTIGUOUS']: image = np.ascontiguousarray(image)
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        if not instances.has("pred_masks") or not instances.has("pred_classes"):
             return np.array([]), np.array([]), np.array([])
        masks = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy() if instances.has("scores") else np.array([])
        return masks, classes, scores
    except Exception as e:
        print(f"Error during Detectron2 inference: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([])


# ===========================
# 🚀 Flask Application Setup
# ===========================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    model_files = [PART_SEG_MODEL_PATH, DAMAGE_SEG_MODEL_PATH, SEVERITY_CLASS_MODEL_PATH]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    status_message = "Car Damage Estimation API is running."
    overall_status = "OK"
    if missing_models:
        status_message += f" WARNING: Models missing: {', '.join(missing_models)}."
        overall_status = "WARNING: Models Missing"
    return jsonify({"message": status_message, "status": overall_status}), 200 if overall_status == "OK" else 500


@app.route('/predict', methods=['POST'])
def predict_damage_cost():
    print("\nReceived request on /predict")
    start_req_time = time.time()
    
    if 'part_predictor' not in globals() or 'damage_predictor' not in globals():
        return jsonify({"error": "Server internal error: Models not loaded."}), 500
    
    if 'image_file' not in request.files:
        return jsonify({"error": "Missing 'image_file' in request"}), 400
        
    file = request.files['image_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None: raise ValueError("Could not decode image.")
        
        image_bgr = resize_large_image(image_bgr, max_dimension=4000)
    except Exception as e:
        print(f"Error reading image: {e}")
        return jsonify({"error": f"Failed to read image: {e}"}), 400

    try:
        # --- THREAD-SAFE INFERENCE BLOCKS ---
        print("Running Part Segmentation...")
        with model_lock:
            part_masks, part_class_idxs, part_scores = run_mask_rcnn(image_bgr, part_predictor)
        
        print(f"Part Segmentation found {len(part_masks)} masks.")
        
        print("Running Damage Segmentation...")
        with model_lock:
            all_damage_masks, all_damage_class_idxs, all_damage_scores = run_mask_rcnn(image_bgr, damage_predictor)
        # ------------------------------------

        # Filter background
        if all_damage_masks.size > 0:
            non_background = all_damage_class_idxs != 0
            damage_masks = all_damage_masks[non_background]
            damage_class_idxs = all_damage_class_idxs[non_background] - 1
            damage_scores = all_damage_scores[non_background] if all_damage_scores.size > 0 else np.array([])
        else:
            damage_masks, damage_class_idxs, damage_scores = np.array([]), np.array([]), np.array([])

        # Label Mapping
        damage_type_labels = [damage_segmentation_class_names[i] if 0 <= i < len(damage_segmentation_class_names) else "Unknown" for i in damage_class_idxs]
        part_labels = [car_part_classes[i] if 0 <= i < len(car_part_classes) else "Unknown" for i in part_class_idxs]

        # Calculate Overlaps
        damages = []
        if damage_masks.size > 0 and part_masks.size > 0:
            for i, dmg_mask in enumerate(damage_masks):
                damage_type = damage_type_labels[i]
                damage_conf = float(damage_scores[i]) if i < len(damage_scores) else None
                
                for j, part_mask in enumerate(part_masks):
                    iou = compute_iou(dmg_mask, part_mask)
                    if iou > COST_ESTIMATION_IOU_THRESHOLD:
                        part_name = part_labels[j]
                        part_conf = float(part_scores[j]) if j < len(part_scores) else None
                        
                        # Bounding Box Logic
                        overlap_mask = np.logical_and(dmg_mask.astype(bool), part_mask.astype(bool))
                        rows, cols = np.where(overlap_mask)
                        if rows.size > 0:
                            y_min, y_max = int(np.min(rows)), int(np.max(rows))
                            x_min, x_max = int(np.min(cols)), int(np.max(cols))
                            padding = 5
                            img_h, img_w = image_bgr.shape[:2]
                            box = [
                                max(0, x_min - padding), max(0, y_min - padding),
                                min(img_w, x_max + 1 + padding), min(img_h, y_max + 1 + padding)
                            ]
                            
                            damages.append({
                                "damage_type": damage_type,
                                "confidence": damage_conf,
                                "damaged_part": part_name,
                                "part_confidence": part_conf,
                                "bounding_box": box
                            })

        # Post-processing
        damages = deduplicate_damages_per_part(damages)
        damages = apply_logical_consistency_filter(damages)

        print("Determining overall severity...")
        # Optional: Lock classifier if memory is extremely tight, but usually safe to run concurrent
        # with model_lock: 
        overall_severity_index = run_yolo_classifier(image_bgr, classifier_session)
        
        overall_severity_name = severity_names[overall_severity_index] if 0 <= overall_severity_index < len(severity_names) else "Unknown"
        overall_severity_name = validate_severity_consistency(damages, overall_severity_name)

        final_result = {
            "overall_severity": overall_severity_name,
            "damages": damages
        }
        
        print(f"Request processed in {time.time() - start_req_time:.2f}s")
        return jsonify(final_result), 200

    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal processing error", "details": str(e)}), 500

if __name__ == '__main__':
    flask_port = int(os.getenv("PORT", 5001))
    print(f"Starting Flask server on port {flask_port}")
    app.run(host="0.0.0.0", port=flask_port, debug=False)