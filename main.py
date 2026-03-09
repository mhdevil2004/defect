from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import numpy as np
from PIL import Image
import uuid
from datetime import datetime

# ── TensorFlow / Keras (graceful degradation) ─────────────────────────────────
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
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
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
}}) 

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
DEFAULT_INPUT_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "DEFECT")

# Global State
_model = None
_model_loaded = False
_load_error = None

def load_model_internal():
    global _model, _model_loaded, _load_error
    if _model_loaded:
        return _model
    
    print(f"[*] Loading model from {MODEL_PATH}...")
    if not _TF_AVAILABLE or _load_model is None:
        _load_error = "TensorFlow/Keras environment not detected. Check requirements.txt."
        print(f"[!] {_load_error}")
        _model_loaded = True
        return None

    if not os.path.exists(MODEL_PATH):
        _load_error = f"Model file missing at {MODEL_PATH}. Current Dir: {os.getcwd()}"
        print(f"[!] {_load_error}")
        _model_loaded = True
        return None

    try:
        # standard fix for Keras 3 vs 2 deserialization errors
        # We explicitly map InputLayer to ensure H5 files load correctly
        custom_objects = {}
        if _InputLayer is not None:
            custom_objects["InputLayer"] = _InputLayer
        
        # Load without compilation for speed and stability
        _model = _load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
        _model_loaded = True
        _load_error = None
        print("[+] Model loaded successfully with custom objects mapping.")
    except Exception as e:
        _load_error = f"Keras load_model failed: {str(e)}"
        print(f"[!] {_load_error}")
        _model = None
        _model_loaded = True
    
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
        
        model = load_model_internal()
        if model is None:
            return jsonify({
                "error": "Engine Offline",
                "details": _load_error or "Unknown initialization error",
                "status": "fail",
                "demo_mode_disabled": True
            }), 503

        try:
            try:
                shape = model.input_shape
                target = (shape[1], shape[2])
            except:
                target = DEFAULT_INPUT_SIZE
            
            arr = _preprocess(img, target)
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
    load_model_internal()
    return jsonify({
        "status": "ok" if _model else "error",
        "model_loaded": _model is not None,
        "load_error": _load_error,
        "demo_mode": False,
        "version": "3.1.5",
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
        "model_loaded": _model_loaded,
        "demo_mode": False,
        "uptime_seconds": 0,
        "app_version": "3.1.0",
        "cpu_percent": 0,
        "ram_percent": 0
    })

@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    if request.method == "DELETE":
        return jsonify({"cleared": 0})
    return jsonify([])

if __name__ == "__main__":
    load_model_internal()
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
