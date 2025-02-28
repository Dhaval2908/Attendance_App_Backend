import traceback
from flask import Flask, Response, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import os
import json
from mtcnn import MTCNN
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process image and extract embeddings using MTCNN and ArcFace
def get_face_embedding(image_path):
    try:
        # Initialize MTCNN face detector
        detector = MTCNN()

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not read image file"

        # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces = detector.detect_faces(img_rgb)

        if len(faces) == 0:
            return None  # No face detected

        # For simplicity, we'll just use the first detected face (if multiple faces are detected)
        x, y, w, h = faces[0]['box']

        # Crop the face from the image
        cropped_face = img_rgb[y:y+h, x:x+w]

        # Extract face embeddings using DeepFace (ArcFace model)
        embeddings = DeepFace.represent(cropped_face, model_name="ArcFace", enforce_detection=False)

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

@app.route("/process_face", methods=["POST"])
def process_face():
    try:
        data = request.json
        image_name = data.get("image_name")
        stored_embedding = data.get("stored_embedding")

        if not image_name or not stored_embedding:
            return jsonify({"error": "Image name and stored embedding are required!"}), 400

        # Convert stored embedding (from Node.js) to a NumPy array
        stored_embedding = np.array(json.loads(stored_embedding))

        # Path to the stored image in Node.js uploads folder
        image_path = os.path.join("uploads", image_name)

        if not os.path.exists(image_path):
            return jsonify({"error": f"Image '{image_name}' not found!"}), 404

        # Extract face embedding from the image
        face_embedding = get_face_embedding(image_path)

        if face_embedding is None:
            return jsonify({"error": "No face detected in the image!"}), 400

        face_embedding = np.array(face_embedding)
        print("✅ Face Embedding Extracted:", face_embedding)


        # Ensure embeddings are valid
        if np.linalg.norm(stored_embedding) == 0 or np.linalg.norm(face_embedding) == 0:
            print("❌ Error: Invalid embeddings! Cannot compute similarity.")
            return
        # Calculate Cosine Similarity
        cosine_similarity = np.dot(stored_embedding, face_embedding) / (
            np.linalg.norm(stored_embedding) * np.linalg.norm(face_embedding)
        )
        # Define similarity threshold
        similarity_threshold = 0.65  
        is_match = cosine_similarity > similarity_threshold

        print("\n🔍 Face Comparison Result:")
        print(f"✅ Similarity Score: {cosine_similarity:.4f}")
        print(f"✅ Match Found: {is_match}")

        return jsonify({"message": bool(is_match)}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
