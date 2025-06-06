from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import csv
import io
import json
from pathlib import Path

app = Flask(__name__)

# Output directory setup
OUTPUT_DIR = Path('data/ocr_training_data/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GT_PATH = OUTPUT_DIR / 'gt.txt'

# Load existing ground truth into memory
ground_truths = {}
if GT_PATH.exists():
    with GT_PATH.open('r') as gt_file:
        reader = csv.reader(gt_file, delimiter='\t')
        ground_truths = {rows[0]: rows[1] for rows in reader if len(rows) == 2}

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'annotation' not in request.form:
        return jsonify({'error': 'Image and annotation JSON are required'}), 400

    image_file = request.files['image']
    annotation_json = request.form['annotation']

    try:
        annotations = json.loads(annotation_json)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400

    if not annotations or not isinstance(annotations, list) or not annotations[0].get('points'):
        return jsonify({'error': 'Invalid annotation format'}), 400

    # Use only the first annotation
    annotation = annotations[0]
    transcription = annotation.get('transcription', '').strip()

    if not transcription or not any(char.isdigit() for char in transcription) or '.' not in transcription:
        return jsonify({'error': 'Transcription must include digits and a decimal point'}), 400

    # Generate a unique filename
    file_name = image_file.filename
    image_path = OUTPUT_DIR / file_name

    try:
        img = Image.open(image_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Failed to open image: {str(e)}'}), 500

    # Crop the image using annotation points
    try:
        points = np.array(annotation['points'])
        x_min = int(min(points[:, 0]))
        y_min = int(min(points[:, 1]))
        x_max = int(max(points[:, 0]))
        y_max = int(max(points[:, 1]))

        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        resized_img = cropped_img.resize((200, 31), Image.LANCZOS)

        # Save cropped image
        resized_img.save(image_path, format='PNG')

        # Update ground truth
        ground_truths[file_name] = transcription
        with GT_PATH.open('w') as gt_file:
            writer = csv.writer(gt_file, delimiter='\t')
            for fname, label in ground_truths.items():
                writer.writerow([fname, label])

        return jsonify({'message': 'Image and transcription saved successfully.'}), 200

    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)