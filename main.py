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

# Try different import strategies
def import_tensorflow():
    """Try multiple ways to import tensorflow"""
    
    # Strategy 1: Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    try:
        # Strategy 1: Direct import
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        logger.info("✅ TensorFlow imported directly")
        return tf, load_model, image
    except ImportError as e:
        logger.warning(f"Direct import failed: {e}")
        
        try:
            # Strategy 2: Try tensorflow-cpu
            import tensorflow_cpu as tf
            from tensorflow_cpu.keras.models import load_model
            from tensorflow_cpu.keras.preprocessing import image
            logger.info("✅ TensorFlow-CPU imported")
            return tf, load_model, image
        except ImportError:
            logger.warning("TensorFlow-CPU import failed")
            
            try:
                # Strategy 3: Try keras directly
                import keras
                from keras.models import load_model
                from keras.preprocessing import image
                logger.info("✅ Keras imported directly")
                return keras, load_model, image
            except ImportError as e:
                logger.error(f"All import strategies failed: {e}")
                return None, None, None

# Import TensorFlow
tf, load_model, image = import_tensorflow()

if tf is not None:
    try:
        # Limit TensorFlow memory usage
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except:
            pass
        
        logger.info(f"Checking model path: {MODEL_PATH}")
        logger.info(f"Current directory: {os.getcwd()}")
        
        if os.path.exists(MODEL_PATH):
            logger.info(f"✅ Model file found. Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
            
            # Custom loader to handle batch_shape issue
            class CompatibleInputLayer:
                def __init__(self, **kwargs):
                    if 'batch_shape' in kwargs:
                        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                    return tf.keras.layers.InputLayer(**kwargs)
            
            custom_objects = {
                'InputLayer': CompatibleInputLayer
            }
            
            # Load the model
            logger.info("Loading model with compatibility fixes...")
            start_time = time.time()
            
            try:
                model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
                load_time = time.time() - start_time
                model_load_time = datetime.now().isoformat()
                logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            except Exception as e:
                # Try without custom objects
                logger.warning(f"Loading with custom objects failed: {e}")
                logger.info("Trying without custom objects...")
                model = load_model(MODEL_PATH, compile=False)
                load_time = time.time() - start_time
                model_load_time = datetime.now().isoformat()
                logger.info(f"✅ Model loaded successfully (without custom objects) in {load_time:.2f} seconds")
            
            # Test the model
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
        else:
            logger.error(f"❌ Model file NOT found at: {MODEL_PATH}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            model_load_error = f"Model file not found at {MODEL_PATH}"
            
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        model_load_error = f"Model loading failed: {str(e)}"
else:
    model_load_error = "TensorFlow/Keras not installed. Check requirements.txt"
    logger.error(model_load_error)

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
        'version': '3.0.0',
        'model_loaded': model is not None,
        'model_error': model_load_error,
        'model_load_time': model_load_time,
        'tensorflow_available': tf is not None,
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
        
        # Process predictions based on output shape
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            defect_prob = float(predictions[0][0])
            normal_prob = 1 - defect_prob
        else:
            normal_prob = float(predictions[0][0])
            defect_prob = float(predictions[0][1]) if len(predictions[0]) > 1 else float(predictions[0][0])
        
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
        return jsonify({
            'current_directory': os.getcwd(),
            'files': files,
            'model_exists': os.path.exists(MODEL_PATH),
            'model_loaded': model is not None,
            'model_error': model_load_error,
            'tensorflow_available': tf is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
