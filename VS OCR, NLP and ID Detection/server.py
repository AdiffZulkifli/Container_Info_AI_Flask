from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import re
import easyocr
import numpy as np
import torch
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- MODEL LOADING ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# Load YOLO model (change to your model path)
model = YOLO("best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))

# Regex patterns
id_pattern = r"[A-Z]{4}\d{7}"

CONF_THRESHOLD = 0.3  # Minimum YOLO confidence


# === OCR correction ===
def clean_and_correct_text(text):
    """Fix common OCR mistakes in container IDs."""
    text = text.upper().replace(" ", "")
    corrections = {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B"
    }
    if len(text) >= 4:
        corrected = ""
        for i, ch in enumerate(text):
            if i < 4 and ch in corrections:
                corrected += corrections[ch]
            else:
                corrected += ch
        text = corrected
    return text


# === Container info extraction (fixed) ===
def extract_container_info(text_list):
    """Extract container ID, ISO code, and container type."""
    combined = "".join(text_list).upper().replace(" ", "")
    combined = clean_and_correct_text(combined)

    container_id = None
    iso_code = None
    container_type = None

    # Regex for container ID (4 letters + 7 digits)
    id_match = re.search(r"[A-Z]{4}\d{7}", combined)
    if id_match:
        container_id = id_match.group()
        # Remove ID portion before looking for ISO
        remaining = combined.replace(container_id, "")
    else:
        remaining = combined

    

    return container_id


@app.route('/ocr', methods=['POST'])
def ocr_from_image():
    """API endpoint to run YOLO + OCR on uploaded image (for Flutter or browser)."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)
    file.save(image_path)

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # YOLO detection
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    roi = img.copy()
    annotated_img = img.copy()
    used_box = False

    # Filter detections by confidence
    for (box, conf) in zip(boxes, confs):
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"Container ID ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        roi = img[y1:y2, x1:x2]
        used_box = True
        break  # use first confident detection

    # If no confident detection, use full image
    if not used_box:
        print("⚠️ No YOLO detections above confidence threshold.")
        roi = img.copy()

    # OCR process
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text_results = reader.readtext(gray, detail=0)

    container_id = extract_container_info(text_results)

    # Overlay OCR results on the annotated image
    y_offset = 40
    if container_id:
        cv2.putText(annotated_img, f"ID: {container_id}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40

    cv2.imwrite(result_path, annotated_img)

    return jsonify({
        'raw_text': text_results,
        'container_id': container_id or "Not found",
        'annotated_image': f"/result/{filename}"
    })


@app.route('/result/<filename>')
def result_image(filename):
    """Return annotated image."""
    return send_file(os.path.join(RESULT_FOLDER, filename), mimetype='image/jpeg')


@app.route('/')
def index():
    """Simple HTML test page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
