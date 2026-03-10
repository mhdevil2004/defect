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

# CRITICAL: Force Keras 2 behavior BEFORE importing tensorflow
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import load_model
    
    logger.info(f"✅ TensorFlow {tf.__version__} imported successfully")
    
    # Create custom objects to handle Keras 3 serialization
    class CompatibleConv2D(layers.Conv2D):
        """Conv2D that handles Keras 3 dtype policy"""
        def __init__(self, *args, **kwargs):
            # Remove dtype policy if present
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
                if 'class_name' in kwargs['dtype'] and kwargs['dtype']['class_name'] == 'DTypePolicy':
                    kwargs.pop('dtype')
            super().__init__(*args, **kwargs)
    
    class CompatibleInputLayer(layers.InputLayer):
        """InputLayer that handles batch_shape"""
        def __init__(self, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(**kwargs)
    
    # Comprehensive custom objects dictionary
    custom_objects = {
        'InputLayer': CompatibleInputLayer,
        'Conv2D': CompatibleConv2D,
        'DTypePolicy': None,  # Ignore DTypePolicy
        'float32': 'float32',  # Map float32 dtype
    }
    
    # Also add all standard layers to handle any other Keras 3 artifacts
    for layer_name in dir(layers):
        if not layer_name.startswith('_') and layer_name[0].isupper():
            layer_class = getattr(layers, layer_name)
            if isinstance(layer_class, type) and issubclass(layer_class, layers.Layer):
                custom_objects[layer_name] = layer_class
    
    logger.info("✅ Created compatibility layers")
    
    # Check if model file exists
    logger.info(f"Checking model path: {MODEL_PATH}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"✅ Model file found. Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        # Load the model with multiple strategies
        start_time = time.time()
        load_success = False
        
        # Strategy 1: Load with custom objects and safe_mode=False
        try:
            logger.info("Strategy 1: Loading with custom objects and safe_mode=False...")
            model = load_model(
                MODEL_PATH, 
                compile=False, 
                custom_objects=custom_objects,
                safe_mode=False
            )
            logger.info("✅ Strategy 1 succeeded")
            load_success = True
        except Exception as e1:
            logger.warning(f"Strategy 1 failed: {e1}")
        
        # Strategy 2: Load without compilation
        if not load_success:
            try:
                logger.info("Strategy 2: Loading without compilation...")
                model = load_model(MODEL_PATH, compile=False)
                logger.info("✅ Strategy 2 succeeded")
                load_success = True
            except Exception as e2:
                logger.warning(f"Strategy 2 failed: {e2}")
        
        # Strategy 3: Load with custom objects only
        if not load_success:
            try:
                logger.info("Strategy 3: Loading with custom objects only...")
                model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
                logger.info("✅ Strategy 3 succeeded")
                load_success = True
            except Exception as e3:
                logger.warning(f"Strategy 3 failed: {e3}")
        
        if load_success:
            load_time = time.time() - start_time
            model_load_time = datetime.now().isoformat()
            logger.info(f"✅ Model loaded in {load_time:.2f} seconds")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            # Test the model
            try:
                dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                dummy_output = model.predict(dummy_input, verbose=0)
                logger.info(f"✅ Model test passed. Output shape: {dummy_output.shape}")
            except Exception as test_e:
                logger.warning(f"Model test warning: {test_e}")
        else:
            model_load_error = "All loading strategies failed"
            
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
        'version': '5.0.0',
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
        file_info = {f: os.path.getsize(f) for f in files if os.path.isfile(f)}
        
        return jsonify({
            'current_directory': os.getcwd(),
            'files': file_info,
            'model_exists': os.path.exists(MODEL_PATH),
            'model_loaded': model is not None,
            'model_error': model_load_error,
            'tensorflow_version': tf.__version__ if 'tf' in dir() else None,
            'keras_mode': 'legacy' if os.environ.get('TF_USE_LEGACY_KERAS') == '1' else 'normal'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
