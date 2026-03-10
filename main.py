from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import numpy as np
from PIL import Image
import logging
import sys
import traceback
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"])

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.h5")
DEFAULT_INPUT_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "DEFECT"]

# Global variables
model = None
model_load_error = None
model_load_time = None

logger.info("="*50)
logger.info("STARTING DEFECT DETECTION API")
logger.info("="*50)

# Fix for Keras 3 compatibility - THIS IS THE KEY SOLUTION
try:
    import tensorflow as tf
    
    # Force Keras 2 behavior
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # Now import from tensorflow.keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras import backend as K
    
    logger.info("✅ TensorFlow imported successfully")
    
    # CRITICAL FIX: Create a custom InputLayer that handles batch_shape
    class PatchedInputLayer(InputLayer):
        """InputLayer that converts batch_shape to batch_input_shape"""
        def __init__(self, **kwargs):
            # Convert batch_shape to batch_input_shape if present
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                logger.info(f"Converted batch_shape to batch_input_shape: {kwargs['batch_input_shape']}")
            super().__init__(**kwargs)
    
    # Register the custom layer
    get_custom_objects().update({'InputLayer': PatchedInputLayer})
    
    # Also patch the deserialization directly
    from tensorflow.keras.layers import deserialize as deserialize_layer
    
    original_deserialize = deserialize_layer
    
    def patched_deserialize(config, custom_objects=None):
        """Patch to handle InputLayer with batch_shape"""
        if isinstance(config, dict):
            class_name = config.get('class_name')
            if class_name == 'InputLayer':
                config_config = config.get('config', {})
                if 'batch_shape' in config_config:
                    config_config['batch_input_shape'] = config_config.pop('batch_shape')
                    config['config'] = config_config
                    logger.info("Patched InputLayer config during deserialization")
        
        return original_deserialize(config, custom_objects)
    
    # Apply the patch
    import tensorflow.keras.layers
    tensorflow.keras.layers.deserialize = patched_deserialize
    
    logger.info("✅ Applied InputLayer compatibility patches")
    
    # Check if model file exists
    logger.info(f"Checking model path: {MODEL_PATH}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"✅ Model file found. Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        # Load the model with multiple strategies
        logger.info("Loading model with compatibility fixes...")
        start_time = time.time()
        
        try:
            # Strategy 1: Load with custom objects
            custom_objects = {'InputLayer': PatchedInputLayer}
            model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            logger.info("✅ Model loaded successfully with custom objects")
        except Exception as e1:
            logger.warning(f"Strategy 1 failed: {e1}")
            
            try:
                # Strategy 2: Load without compilation and custom objects
                model = load_model(MODEL_PATH, compile=False)
                logger.info("✅ Model loaded successfully without custom objects")
            except Exception as e2:
                logger.warning(f"Strategy 2 failed: {e2}")
                
                try:
                    # Strategy 3: Load with safe_mode=False
                    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
                    logger.info("✅ Model loaded successfully with safe_mode=False")
                except Exception as e3:
                    logger.error(f"All loading strategies failed: {e3}")
                    raise
        
        load_time = time.time() - start_time
        model_load_time = datetime.now().isoformat()
        
        logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        # Test model with dummy data
        try:
            dummy_input = np.zeros((1, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1], 3), dtype=np.float32)
            dummy_output = model.predict(dummy_input, verbose=0)
            logger.info(f"✅ Model test successful. Output shape: {dummy_output.shape}")
        except Exception as test_e:
            logger.warning(f"Model test warning: {test_e}")
            
    else:
        logger.error(f"❌ Model file NOT found at: {MODEL_PATH}")
        model_load_error = f"Model file not found at {MODEL_PATH}"
        
except ImportError as e:
    logger.error(f"❌ TensorFlow import failed: {str(e)}")
    model_load_error = f"TensorFlow import failed: {str(e)}"
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    logger.error(traceback.format_exc())
    model_load_error = str(e)

logger.info("="*50)

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(DEFAULT_INPUT_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'status': 'online',
        'name': 'Defect Detection API',
        'version': '4.0.0',
        'model_loaded': model is not None,
        'model_error': model_load_error,
        'model_load_time': model_load_time,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect():
    """Main detection endpoint"""
    # Handle preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': f'Model not loaded: {model_load_error}',
            'status': 'model_unavailable'
        }), 503
    
    # Check if image is present
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image uploaded. Please send image with key "image"'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    try:
        # Read image bytes
        img_bytes = file.read()
        
        # Preprocess image
        processed_img = preprocess_image(img_bytes)
        
        # Run inference
        inference_start = time.time()
        predictions = model.predict(processed_img, verbose=0)
        inference_time = time.time() - inference_start
        
        # Process predictions
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # Binary classification
            defect_prob = float(predictions[0][0])
            normal_prob = 1 - defect_prob
        elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
            # Multi-class with 2 outputs
            normal_prob = float(predictions[0][0])
            defect_prob = float(predictions[0][1])
        else:
            # Handle other cases
            defect_prob = float(np.max(predictions[0]))
            normal_prob = 1 - defect_prob
        
        # Determine result
        if defect_prob > 0.5:
            prediction = "defect_detected"
            confidence = defect_prob
            class_name = "DEFECT"
        else:
            prediction = "normal"
            confidence = normal_prob
            class_name = "NORMAL"
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'class': class_name,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'normal': round(normal_prob * 100, 2),
                'defect': round(defect_prob * 100, 2)
            },
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add CORS headers
        response_json = jsonify(response)
        response_json.headers.add('Access-Control-Allow-Origin', '*')
        return response_json, 200
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint"""
    try:
        files = os.listdir('.')
        file_info = {}
        for f in files:
            if os.path.isfile(f):
                file_info[f] = os.path.getsize(f)
        
        return jsonify({
            'current_directory': os.getcwd(),
            'files': file_info,
            'model_exists': os.path.exists(MODEL_PATH),
            'model_loaded': model is not None,
            'model_error': model_load_error,
            'tensorflow_version': tf.__version__ if 'tf' in dir() else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
