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
    from tensorflow.keras.models import load_model as _load_model          # type: ignore
    from tensorflow.keras.preprocessing import image as _keras_image       # type: ignore
    _TF_AVAILABLE = True
except Exception:
    try:
        from keras.models import load_model as _load_model                 # type: ignore
        from keras.preprocessing import image as _keras_image             # type: ignore
        _TF_AVAILABLE = True
    except Exception:
        _load_model = None       # type: ignore
        _keras_image = None      # type: ignore
        _TF_AVAILABLE = False

app = Flask(__name__)
# In production, you might want to restrict this to your Vercel domain
CORS(app, resources={r"/api/*": {"origins": "*"}}) 

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")

DEFAULT_INPUT_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "DEFECT")

# Global State
_model = None
_model_loaded = False

def load_model_internal():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    
    print(f"[*] Loading model from {MODEL_PATH}...")
    if not _TF_AVAILABLE or _load_model is None:
        print("[!] TensorFlow/Keras not available. Running in DEMO mode.")
        _model_loaded = True
        return None

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file not found at {MODEL_PATH}. Running in DEMO mode.")
        _model_loaded = True
        return None

    try:
        # Load without compilation for speed
        _model = _load_model(MODEL_PATH, compile=False)
        _model_loaded = True
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Forceful load failure: {e}. Falling back to DEMO mode.")
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

def _demo_predict(image):
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    brightness = float(np.mean(arr))
    noise = float(np.std(arr))
    defect_score = 0.0
    if brightness < 60 or brightness > 215:
        defect_score += 0.45
    if noise > 70:
        defect_score += 0.35
    defect_score = float(min(max(defect_score + np.random.uniform(-0.05, 0.05), 0.01), 0.99))
    return {"NORMAL": round(1 - defect_score, 4), "DEFECT": round(defect_score, 4)}

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
        demo_mode = model is None

        if demo_mode:
            probs = _demo_predict(img)
        else:
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
        
        return jsonify({
            "prediction": prediction,
            "confidence": round(float(defect_p if prediction == "defect_detected" else probs.get("NORMAL", 1-defect_p)), 2),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    model = load_model_internal()
    return jsonify({
        "status": "ok",
        "model_loaded": _model_loaded,
        "demo_mode": model is None,
        "version": "3.1.0",
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
        "demo_mode": _model is None,
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

