from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import numpy as np
from PIL import Image
import logging
import sys
import threading
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Allow all origins for mobile app compatibility
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
DEFAULT_INPUT_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "DEFECT"]

# Global state
model = None
model_loading = False
model_load_error = None
model_load_lock = threading.Lock()

# TensorFlow/Keras import with fallback
try:
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    import tensorflow as tf
    
    # Limit memory growth and CPU threads for Render
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    
    logger.info("✓ TensorFlow imported successfully")
    TF_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"✗ TensorFlow import failed: {e}")
    TF_AVAILABLE = False

@app.route('/', methods=['GET'])
def home():
    """Root endpoint for health check"""
    return jsonify({
        'status': 'online',
        'message': 'Defect Detection API is running',
        'model_loaded': model is not None,
        'model_loading': model_loading,
        'model_error': model_load_error,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model is not None else 'loading' if model_loading else 'error',
        'error': model_load_error,
        'timestamp': datetime.now().isoformat()
    }), 200

def load_model_async():
    """Load model in background thread"""
    global model, model_loading, model_load_error
    
    with model_load_lock:
        if model is not None or model_loading:
            return
        model_loading = True
    
    logger.info("🔄 Starting model loading...")
    
    try:
        if not TF_AVAILABLE:
            model_load_error = "TensorFlow not available"
            logger.error("✗ TensorFlow not available")
            return
        
        if not os.path.exists(MODEL_PATH):
            model_load_error = f"Model file not found at {MODEL_PATH}"
            logger.error(f"✗ {model_load_error}")
            logger.info(f"Current directory: {os.getcwd()}")
            logger.info(f"Files: {os.listdir('.')}")
            return
        
        logger.info(f"📁 Loading model from: {MODEL_PATH}")
        start_time = time.time()
        
        # Simple model loading without custom objects
        model = load_model(MODEL_PATH, compile=False)
        
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        model_load_error = None
        
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"✗ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        model = None
        
    finally:
        model_loading = False

def preprocess_image(image_file, target_size=DEFAULT_INPUT_SIZE):
    """Preprocess image for model inference"""
    try:
        # Read and convert image
        img = Image.open(io.BytesIO(image_file.read()))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        logger.info(f"✓ Image preprocessed: shape={img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"✗ Image preprocessing failed: {e}")
        raise

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect_defect():
    """Main detection endpoint"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check if model is ready
    if model is None:
        if model_loading:
            return jsonify({
                'success': False,
                'error': 'Model is still loading. Please try again in a few seconds.',
                'status': 'loading'
            }), 503
        else:
            # Start loading in background
            threading.Thread(target=load_model_async, daemon=True).start()
            return jsonify({
                'success': False,
                'error': 'Model is being initialized. Please try again in 30 seconds.',
                'status': 'initializing'
            }), 503
    
    # Check if image is present
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    try:
        # Preprocess image
        processed_image = preprocess_image(file)
        
        # Run inference
        logger.info("🔍 Running inference...")
        start_time = time.time()
        
        predictions = model.predict(processed_image, verbose=0)
        
        inference_time = time.time() - start_time
        logger.info(f"✓ Inference completed in {inference_time*1000:.2f}ms")
        
        # Process predictions
        if predictions.shape[-1] == 1:
            # Binary classification with sigmoid
            defect_prob = float(predictions[0][0])
            normal_prob = 1 - defect_prob
            probs = [normal_prob, defect_prob]
        else:
            # Multi-class with softmax
            probs = predictions[0].tolist()
        
        # Determine class
        predicted_class_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probs[predicted_class_idx]
        
        # Format response
        response = {
            'success': True,
            'prediction': 'defect_detected' if predicted_class == 'DEFECT' else 'normal',
            'confidence': round(confidence * 100, 2),  # Return as percentage
            'class': predicted_class,
            'probabilities': {
                'normal': round(probs[0] * 100, 2),
                'defect': round(probs[1] * 100, 2)
            },
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✓ Prediction: {response['prediction']} (confidence: {response['confidence']}%)")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"✗ Detection failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'loaded': False,
            'loading': model_loading,
            'error': model_load_error
        }), 200
    
    try:
        return jsonify({
            'loaded': True,
            'loading': False,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'classes': CLASS_NAMES,
            'error': None
        }), 200
    except Exception as e:
        return jsonify({
            'loaded': True,
            'error': str(e)
        }), 200

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'success': True,
        'message': 'API is working correctly',
        'model_status': 'loaded' if model is not None else 'loading' if model_loading else 'not loaded',
        'timestamp': datetime.now().isoformat()
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Start model loading in background
    logger.info("🚀 Starting Defect Detection API...")
    threading.Thread(target=load_model_async, daemon=True).start()
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 8000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
