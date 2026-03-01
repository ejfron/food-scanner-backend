# backend-easyocr/app.py
import base64
import logging
import re
import numpy as np
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Initialize EasyOCR reader (only once)
reader = easyocr.Reader(['en'])

@app.route('/ocr/nutrition', methods=['POST'])
def extract_nutrition():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Convert PIL Image to numpy array (EasyOCR expects numpy array)
        image_np = np.array(image)

        # Run EasyOCR
        result = reader.readtext(image_np, detail=0, paragraph=True)
        full_text = ' '.join(result)

        logging.info(f"OCR text: {full_text[:500]}...")

        # Return raw text so your frontend parser can handle it
        return jsonify({
            'ocrText': full_text,
            'confidence': 0.9
        })

    except Exception as e:
        logging.exception("OCR failed")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)