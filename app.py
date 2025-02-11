from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STITCHED_FOLDER = 'stitched'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STITCHED_FOLDER):
    os.makedirs(STITCHED_FOLDER)

def stitch_images(images):
    # Stitch images using OpenCV
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise Exception('Image stitching failed!')
    return stitched

def reconstruct_lost_parts(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a binary mask where the non-black pixels are set to 255 (white)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Invert the mask to get the lost parts (black areas in the stitched image)
    mask = cv2.bitwise_not(mask)
    # Inpaint the image using the mask
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('images')
    images = []
    for file in files:
        # Save uploaded files
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        # Read images using OpenCV
        image = cv2.imread(filename)
        images.append(image)

    try:
        # Stitch images
        stitched = stitch_images(images)
        # Save the stitched image
        stitched_filename = os.path.join(STITCHED_FOLDER, 'stitched.png')
        cv2.imwrite(stitched_filename, stitched)

        return send_file(stitched_filename, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reconstruct', methods=['GET'])
def reconstruct_image():
    try:
        stitched_filename = os.path.join(STITCHED_FOLDER, 'stitched.png')
        stitched_image = cv2.imread(stitched_filename)
        reconstructed = reconstruct_lost_parts(stitched_image)
        reconstructed_filename = os.path.join(STITCHED_FOLDER, 'reconstructed.png')
        cv2.imwrite(reconstructed_filename, reconstructed)
        return send_file(reconstructed_filename, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
