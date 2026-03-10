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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
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

# Import TensorFlow with proper settings
logger.info("="*50)
logger.info("STARTING DEFECT DETECTION API")
logger.info("="*50)

try:
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    logger.info("Importing TensorFlow...")
    import tensorflow as tf
    
    # Limit TensorFlow memory usage
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    
    logger.info("✅ TensorFlow imported successfully")
    
    # Check if model file exists
    logger.info(f"Checking model path: {MODEL_PATH}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"✅ Model file found at: {MODEL_PATH}")
        logger.info(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        # Load the model
        logger.info("Loading model...")
        start_time = time.time()
        
        # Simple loading without custom objects
        model = load_model(MODEL_PATH, compile=False)
        
        load_time = time.time() - start_time
        model_load_time = datetime.now().isoformat()
        
        logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        # Test with dummy data
        logger.info("Testing model with dummy data...")
        dummy_input = np.zeros((1, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1], 3))
        dummy_output = model.predict(dummy_input, verbose=0)
        logger.info(f"✅ Model test successful. Output shape: {dummy_output.shape}")
        
    else:
        logger.error(f"❌ Model file NOT found at: {MODEL_PATH}")
        logger.error(f"Current directory absolute path: {os.path.abspath('.')}")
        logger.error(f"All files in current directory: {os.listdir('.')}")
        model_load_error = f"Model file not found at {MODEL_PATH}"
        
except ImportError as e:
    logger.error(f"❌ TensorFlow import failed: {str(e)}")
    model_load_error = f"TensorFlow import failed: {str(e)}"
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    logger.error(traceback.format_exc())
    model_load_error = f"Model loading failed: {str(e)}"

logger.info("="*50)

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Original image: mode={img.mode}, size={img.size}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted to RGB")
        
        # Resize
        img = img.resize(DEFAULT_INPUT_SIZE, Image.Resampling.LANCZOS)
        logger.info(f"Resized to: {img.size}")
        
        # Convert to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        logger.info(f"Final array shape: {img_array.shape}")
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
        'version': '1.0.0',
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
        return '', 200
    
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
            'error': 'No image uploaded'
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
        logger.info(f"Received image: {file.filename}, size: {len(img_bytes)} bytes")
        
        # Preprocess image
        processed_img = preprocess_image(img_bytes)
        
        # Run inference
        logger.info("Running inference...")
        inference_start = time.time()
        
        predictions = model.predict(processed_img, verbose=0)
        
        inference_time = time.time() - inference_start
        logger.info(f"Inference completed in {inference_time*1000:.2f}ms")
        logger.info(f"Raw predictions: {predictions}")
        
        # Process predictions
        if predictions.shape[-1] == 1:
            # Binary classification (sigmoid output)
            defect_prob = float(predictions[0][0])
            normal_prob = 1 - defect_prob
            logger.info(f"Binary output - defect: {defect_prob:.4f}, normal: {normal_prob:.4f}")
        else:
            # Multi-class (softmax output)
            defect_prob = float(predictions[0][1]) if len(predictions[0]) > 1 else float(predictions[0][0])
            normal_prob = float(predictions[0][0])
        
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
        
        logger.info(f"✅ Prediction: {prediction} (confidence: {response['confidence']}%)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to list files"""
    try:
        files = os.listdir('.')
        file_details = []
        for f in files:
            if os.path.isfile(f):
                file_details.append({
                    'name': f,
                    'size': os.path.getsize(f),
                    'modified': datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
                })
        
        return jsonify({
            'current_directory': os.getcwd(),
            'files': file_details,
            'model_path': MODEL_PATH,
            'model_exists': os.path.exists(MODEL_PATH),
            'model_loaded': model is not None,
            'model_error': model_load_error
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
