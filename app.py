from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import os

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process image and extract embeddings using ArcFace
def get_face_embedding(image_path):
    try:
        # Open image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not read image file"

        # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract face embeddings using ArcFace
        embeddings = DeepFace.represent(img_rgb, model_name="ArcFace", enforce_detection=False)

        # If no face found, return None
        if not embeddings:
            return None

        return embeddings[0]["embedding"]
    
    except Exception as e:
        return str(e)

# API Route: Register a face (upload + extract embedding)
@app.route("/register", methods=["POST"])
def register_student():
    try:
        # Get image from request
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "Image is required!"}), 400

        # Save image to uploads folder
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Process image to get face embedding using ArcFace
        embeddings = get_face_embedding(image_path)
        if embeddings is None:
            return jsonify({"error": "No face detected in the image!"}), 400

        return jsonify({"message": "Face embedding extracted successfully!", "face_embedding": embeddings}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
