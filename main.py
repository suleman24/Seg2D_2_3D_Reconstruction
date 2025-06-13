import os
import uuid
import logging
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from segment_anything import sam_model_registry, SamPredictor
from trellis import generate_glb_from_image  # Import the function from trellis.py

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create directory for uploads if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

if not os.path.isfile(sam_checkpoint):
    raise FileNotFoundError(f"Checkpoint file {sam_checkpoint} not found")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    logger.debug("Received request to /segment")

    if 'image' not in request.files or 'x' not in request.form or 'y' not in request.form:
        return jsonify({'error': 'Missing image or coordinates'}), 400

    file = request.files['image']
    x = int(float(request.form['x']))
    y = int(float(request.form['y']))

    # Save original image
    original_filename = f"{uuid.uuid4()}.jpg"
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(original_path)

    # Load and convert image
    image = cv2.imread(original_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image in predictor
    predictor.set_image(image)

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Use the mask with the highest score
    best_mask = masks[np.argmax(scores)]
    padding = 100

    # Find bounding box of the segmented object
    ys, xs = np.where(best_mask)
    if len(xs) == 0 or len(ys) == 0:
        return jsonify({'error': 'No valid segmentation found'}), 400

    x_min, x_max = max(xs.min() - padding, 0), min(xs.max() + padding, image.shape[1] - 1)
    y_min, y_max = max(ys.min() - padding, 0), min(ys.max() + padding, image.shape[0] - 1)

    # Crop the original image and mask
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = best_mask[y_min:y_max+1, x_min:x_max+1]

    # Apply mask to isolate the object
    cropped_mask = cropped_mask.astype(np.uint8) * 255
    segmented_object = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    # Ensure non-masked areas are transparent
    if segmented_object.shape[2] == 3:
        alpha_channel = np.where(cropped_mask == 255, 255, 0).astype(np.uint8)
        segmented_object = np.dstack((segmented_object, alpha_channel))

    # Save the cropped segmented object
    segmented_filename = f"segmented_{uuid.uuid4()}.png"
    segmented_path = os.path.join(UPLOAD_FOLDER, segmented_filename)
    cv2.imwrite(segmented_path, cv2.cvtColor(segmented_object, cv2.COLOR_RGBA2BGRA if segmented_object.shape[2] == 4 else cv2.COLOR_RGB2BGR))

    # Generate GLB file using trellis.py
    try:
        # glb_path = generate_glb_from_image(segmented_path)
        # glb_url = glb_path
        glb_url = 'static/uploads/model_8c301ddc-1968-4564-b664-38292427942a.glb'
        print('********')
        print(glb_url)
        print('********')
    except Exception as e:
        logger.error(f"Error generating GLB: {str(e)}")
        glb_url = '/static/uploads/model.glb'

    return jsonify({
        'segmented_image': f'/{segmented_path}',
        'original_image': f'/{original_path}',
        'glb_file': glb_url
    })

if __name__ == '__main__':
    app.run(debug=True, port=5005)