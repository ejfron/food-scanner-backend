# backend-easyocr/app.py
import base64
import logging
import time
import os
from io import BytesIO

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_compress import Compress
from PIL import Image
import pytesseract

# ---------------------------
# Configuration
# ---------------------------
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_PIXELS = 2000   # resize larger images to this max dimension
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# ---------------------------
# App Initialization
# ---------------------------
app = Flask(__name__)

# Enable Gzip compression for responses (smaller/faster downloads)
Compress(app)

# Flexible CORS – allows your Render domain, local dev, Capacitor, and dev tunnels
# You can also set CORS_ORIGINS environment variable to add more origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:8080",
    "capacitor://localhost",
    "https://*.render.com",
    "https://*.asse.devtunnels.ms",   # Azure dev tunnels
    "file://",                         # for APK file access (Android)
]
# If you need to allow any origin during development, set env ALLOW_CORS_ALL=true
if os.environ.get("ALLOW_CORS_ALL") == "true":
    CORS(app, origins="*")
else:
    CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Utility Functions
# ---------------------------
def resize_image_if_needed(image, max_dimension=MAX_IMAGE_PIXELS):
    """Resize image if either dimension exceeds max_dimension, preserving aspect ratio."""
    if max(image.size) <= max_dimension:
        return image
    ratio = max_dimension / float(max(image.size))
    new_size = tuple(int(dim * ratio) for dim in image.size)
    return image.resize(new_size, Image.Resampling.LANCZOS)

def process_ocr(image_bytes):
    """Run OCR on image bytes, with resizing and timing."""
    try:
        # Load image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        original_size = img.size
        logger.info(f"Original image size: {original_size}")

        # Resize if too large
        img = resize_image_if_needed(img)
        if img.size != original_size:
            logger.info(f"Resized to: {img.size}")

        # Run Tesseract
        start_ocr = time.time()
        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        ocr_time = time.time() - start_ocr
        logger.info(f"OCR completed in {ocr_time:.2f}s, text length: {len(text)}")
        return text, ocr_time
    except Exception as e:
        logger.exception("OCR processing failed")
        raise

# ---------------------------
# Routes
# ---------------------------
@app.route('/health', methods=['GET'])
def health():
    """Simple health check – useful for keep‑alive pings."""
    return jsonify({'status': 'ok', 'timestamp': time.time()})

@app.route('/ocr/nutrition', methods=['POST', 'OPTIONS'])
def extract_nutrition():
    """
    Endpoint that accepts a base64 image and returns OCR text.
    Handles CORS preflight automatically (due to flask_cors).
    """
    # Handle preflight manually if needed (flask_cors usually does it, but we keep it safe)
    if request.method == 'OPTIONS':
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
        # Remove potential data URL prefix (e.g., "data:image/jpeg;base64,")
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
        # You could add confidence estimation here (optional)
        return jsonify({
            'ocrText': text,
            'confidence': 0.9,  # placeholder, you can compute real confidence
            'processingTime': ocr_time
        })
    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({'error': 'OCR processing error'}), 500

# ---------------------------
# Run (for development only)
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Use threaded=True to handle multiple requests concurrently
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)