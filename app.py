# backend-easyocr/app.py
import base64
import logging
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image

app = Flask(__name__)
# Allow Capacitor origins and your Render domain
CORS(app, origins=[
    "capacitor://localhost",
    "http://localhost",
    "https://*.render.com"
])
logging.basicConfig(level=logging.INFO)

TESSERACT_CONFIG = r'--oem 3 --psm 6'

@app.route('/ocr/nutrition', methods=['POST'])
def extract_nutrition():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        full_text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
        logging.info(f"OCR text: {full_text[:500]}...")
        return jsonify({'ocrText': full_text, 'confidence': 0.9})
    except Exception as e:
        logging.exception("OCR failed")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)