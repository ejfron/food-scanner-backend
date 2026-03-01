# backend-easyocr/app.py
import base64
import logging
import re
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Tesseract configuration (you can adjust these)
# --oem 3 uses default LSTM engine, --psm 6 treats image as a uniform block of text
TESSERACT_CONFIG = r'--oem 3 --psm 6'

@app.route('/ocr/nutrition', methods=['POST'])
def extract_nutrition():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Run Tesseract OCR
        # pytesseract can work directly with PIL Image objects
        full_text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)

        logging.info(f"OCR text: {full_text[:500]}...")

        # Return raw text (same format as before)
        return jsonify({
            'ocrText': full_text,
            'confidence': 0.9   # Tesseract doesn't provide confidence, so we keep a static value
        })

    except Exception as e:
        logging.exception("OCR failed")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render, default to 5001 for local dev
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)