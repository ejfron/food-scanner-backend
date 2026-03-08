# backend-easyocr/app.py
import base64
import logging
import time
import os
from io import BytesIO

from flask import Flask, request, jsonify, make_response
from PIL import Image
import pytesseract

# Optional compression (skip if not installed)
try:
    from flask_compress import Compress
    compress = Compress()
    HAS_COMPRESS = True
except ImportError:
    HAS_COMPRESS = False
    compress = None

# ---------------------------
# Configuration
# ---------------------------
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_PIXELS = 2000
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# ---------------------------
# App Initialization
# ---------------------------
app = Flask(__name__)

# Enable Gzip compression if available
if HAS_COMPRESS:
    compress.init_app(app)
    app.config['COMPRESS_ALGORITHM'] = 'gzip'
    app.config['COMPRESS_LEVEL'] = 6
    app.config['COMPRESS_MIN_SIZE'] = 500
    logging.info("Flask-Compress enabled")
else:
    logging.warning("Flask-Compress not installed – responses will not be compressed")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# CORS Configuration (Manual)
# ---------------------------
def is_origin_allowed(origin):
    """Return True if the origin is allowed to access the API."""
    allowed_exact = [
        "http://localhost:5173",
        "http://localhost:8080",
        "capacitor://localhost",
        "file://",
    ]
    if origin in allowed_exact:
        return True

    if origin:
        # Allow any subdomain of asse.devtunnels.ms or render.com
        if origin.endswith(".asse.devtunnels.ms") or origin.endswith(".render.com"):
            return True

    return False

@app.after_request
def add_cors_headers(response):
    """Set CORS headers for every response."""
    origin = request.headers.get('Origin')
    if origin and is_origin_allowed(origin):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/ocr/nutrition', methods=['POST', 'OPTIONS'])
def extract_nutrition():
    """Handle preflight OPTIONS request."""
    if request.method == 'OPTIONS':
        # Preflight request – just return headers
        return '', 200

    # Log request metadata
    logger.info(f"Request from: {request.remote_addr}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Content-Length: {request.content_length}")

    # Enforce max size
    if request.content_length and request.content_length > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        logger.warning(f"Request too large: {request.content_length} bytes")
        return jsonify({'error': f'Image too large. Max {MAX_IMAGE_SIZE_MB}MB'}), 413

    # Parse JSON
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.exception("Invalid JSON")
        return jsonify({'error': 'Invalid JSON'}), 400

    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64
    try:
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]
        image_bytes = base64.b64decode(image_b64)
        logger.info(f"Decoded {len(image_bytes)} bytes")
    except Exception as e:
        logger.exception("Base64 decode failed")
        return jsonify({'error': 'Invalid image encoding'}), 400

    # Process OCR
    try:
        text, ocr_time = process_ocr(image_bytes)
        return jsonify({
            'ocrText': text,
            'confidence': 0.9,
            'processingTime': ocr_time
        })
    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({'error': 'OCR processing error'}), 500

def process_ocr(image_bytes):
    """Run OCR with resizing and timing."""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    original_size = img.size
    logger.info(f"Original image size: {original_size}")

    # Resize if too large
    if max(img.size) > MAX_IMAGE_PIXELS:
        ratio = MAX_IMAGE_PIXELS / float(max(img.size))
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized to: {img.size}")

    start_ocr = time.time()
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    ocr_time = time.time() - start_ocr
    logger.info(f"OCR completed in {ocr_time:.2f}s, text length: {len(text)}")
    return text, ocr_time

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)