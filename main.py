from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
import threading

# ── TensorFlow / Keras (graceful degradation) ─────────────────────────────────
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Force Keras 2 if Keras 3 is present
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf
    # Limit memory and CPU usage for Render free tier
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except: pass

    _load_model = tf.keras.models.load_model
    _keras_image = tf.keras.preprocessing.image
    from tensorflow.keras.layers import InputLayer as _InputLayer
    _TF_AVAILABLE = True
except Exception:
    try:
        import keras as _k
        _load_model = _k.models.load_model
        _keras_image = _k.preprocessing.image
        from keras.layers import InputLayer as _InputLayer
        _TF_AVAILABLE = True
    except Exception:
        _load_model = None       # type: ignore
        _keras_image = None      # type: ignore
        _InputLayer = None       # type: ignore
        _TF_AVAILABLE = False

app = Flask(__name__)
# Permissive CORS for broad mobile compatibility
CORS(app) 

@app.route("/")
def index():
    return jsonify({
        "message": "DefectVision API is active",
        "engine_status": "ready" if _model else ("loading" if _model_loading else "offline"),
        "version": "4.1.0"
    }), 200

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
DEFAULT_INPUT_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "DEFECT")

# Global State
_model = None
_model_loading = False
_model_loaded = False
_load_error = None
_state_lock = threading.Lock()

def load_model_internal():
    global _model, _model_loaded, _load_error, _model_loading
    with _state_lock:
        if _model_loaded and _model:
            return _model
        if _model_loading:
            return None
        _model_loading = True
    
    print(f"[*] Loading model from {MODEL_PATH} (Async)...")
    if not _TF_AVAILABLE or _load_model is None:
        _load_error = "TensorFlow/Keras environment not detected. Check requirements.txt and Render logs for installation errors."
        print(f"[!] {_load_error}")
        _model_loaded = True
        _model_loading = False
        return None

    if not os.path.exists(MODEL_PATH):
        _load_error = f"Model file missing at {MODEL_PATH}. Current Dir: {os.getcwd()}. Files: {os.listdir('.')}"
        print(f"[!] {_load_error}")
        _model_loaded = True
        _model_loading = False
        return None

    try:
        # Standard fix for Keras 3 vs 2 deserialization errors
        
        # Dummy class to handle Keras 3 DTypePolicy payloads in Keras 2
        class DummyDTypePolicy:
            def __init__(self, name="float32", **kwargs):
                if isinstance(name, dict) and "name" in name:
                    name = name["name"]
                self.name = name
                self.compute_dtype = name
                self.variable_dtype = name
            def __str__(self):
                return self.name
            def as_list(self):
                return [self.name]

        # Patch custom objects to intercept the unknown DTypePolicy class
        custom_objects = {"DTypePolicy": DummyDTypePolicy}
        
        if _InputLayer is not None:
            # Patch InputLayer for Keras 3 which rejects "batch_shape"
            class PatchedInputLayer(_InputLayer):
                def __init__(self, **kwargs):
                    if "batch_shape" in kwargs:
                        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
                    super().__init__(**kwargs)
            custom_objects["InputLayer"] = PatchedInputLayer
        
        # Add Functional to handle newer Keras 3 structural requirements
        try:
            from tensorflow.keras.models import Model
            custom_objects["Functional"] = Model
            custom_objects["Model"] = Model
        except Exception as e:
            print(f"[*] Note: Could not inject Model/Functional into custom_objects: {e}")

        # Start timer for loading
        start_t = time.time()
        # Load without compilation for speed and stability on limited memory
        _model = _load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
        _model_loaded = True
        _load_error = None
        duration = round(time.time() - start_t, 2)
        print(f"[+] Model loaded successfully in {duration}s with comprehensive custom objects.")
    except Exception as e:
        _load_error = f"Keras load_model final failure: {str(e)}"
        print(f"[!] {_load_error}")
        # Log more detail for OOM or specific Keras errors
        if "MemoryError" in str(e) or "allocate" in str(e).lower():
            print("[!] CRITICAL: Memory limit likely reached on Render.")
        _model = None
        _model_loaded = False 
    finally:
        _model_loading = False
    
    return _model

def _preprocess(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    if _keras_image is not None:
        arr = _keras_image.img_to_array(image)
    else:
        arr = np.array(image, dtype=np.float32)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

# Removed _demo_predict to avoid any fake data generation

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        if not _model and not _model_loading:
            threading.Thread(target=load_model_internal, daemon=True).start()

        model = load_model_internal()
        if model is None:
            return jsonify({
                "error": "Engine Starting",
                "details": "The AI engine is currently initializing in the background. Please wait ~1 minute.",
                "status": "loading",
                "demo_mode_disabled": True
            }), 503

        try:
            # Flexible input shape detection
            try:
                # Try getting shape from model or its first layer
                in_shape = getattr(model, "input_shape", None)
                if not in_shape or not isinstance(in_shape, (list, tuple)) or len(in_shape) < 3:
                     in_shape = model.layers[0].input_shape
                
                # Extract W/H (handles [None, H, W, C] or [H, W, C])
                if len(in_shape) == 4:
                    target = (in_shape[1], in_shape[2])
                else:
                    target = (in_shape[0], in_shape[1])
                
                # Final safety check for None values
                if None in target or not all(isinstance(v, int) for v in target):
                    target = DEFAULT_INPUT_SIZE
            except Exception as shape_e:
                print(f"[*] Fallback to default size: {shape_e}")
                target = DEFAULT_INPUT_SIZE
            
            arr = _preprocess(img, target)
            print(f"[*] Running inference on shape: {arr.shape}")
            raw = model.predict(arr, verbose=0)
            
            if raw.ndim == 2 and raw.shape[1] == 1:
                defect_p = float(raw[0, 0])
                probs = {"NORMAL": round(1 - defect_p, 4), "DEFECT": round(defect_p, 4)}
            else:
                probs = {CLASS_NAMES[i]: round(float(raw[0, i]), 4) for i in range(len(CLASS_NAMES))}

            defect_p = probs.get("DEFECT", 0.0)
            prediction = "defect_detected" if defect_p >= 0.5 else "normal"
            
            print(f"[+] Prediction: {prediction} ({probs})")
            return jsonify({
                "prediction": prediction,
                "confidence": round(float(defect_p if prediction == "defect_detected" else probs.get("NORMAL", 1-defect_p)), 2),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as inner_e:
            print(f"[!] Inner prediction failure: {inner_e}")
            return jsonify({"error": f"Inference engine failure: {str(inner_e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    # Only try loading if not already loaded, not currently loading, and if there's no fatal load error blocking it
    if not _model and not _model_loading and not _load_error:
        threading.Thread(target=load_model_internal, daemon=True).start()
    
    return jsonify({
        "status": "ready" if _model else ("error" if _load_error else "loading"),
        "model_loaded": _model is not None,
        "is_loading": _model_loading,
        "load_error": _load_error,
        "demo_mode": False,
        "version": "4.0.0",
        "uptime_seconds": time.time() - os.path.getmtime(__file__)
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "total": 0,
        "defects": 0,
        "normal": 0,
        "defect_rate": 0,
        "avg_confidence": 0,
        "avg_latency_ms": 0,
        "model_loaded": _model is not None,
        "demo_mode": False,
        "uptime_seconds": time.time() - os.path.getmtime(__file__),
        "app_version": "3.2.0",
        "cpu_percent": 0,
        "ram_percent": 0
    })

@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    if request.method == "DELETE":
        return jsonify({"cleared": 0})
    return jsonify([])

if __name__ == "__main__":
    # Start loading in background to avoid blocking server start
    threading.Thread(target=load_model_internal).start()
    
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
