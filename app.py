import traceback
from flask import Flask, Response, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import os
import json
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MTCNN once
detector = MTCNN()

def crop_face_with_mtcnn(image_path):
    """ Detect and crop the face using MTCNN """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        return None, None  # No face found

    # Use first face detected
    x, y, w, h = faces[0]['box']
    cropped_face = img_rgb[y:y + h, x:x + w]
    return cropped_face, faces[0]


def get_face_embedding(face_image):
    """ Extract embedding using ArcFace on cropped face """
    # Resize to ArcFace input size (112x112)
    face_resized = cv2.resize(face_image, (112, 112))

    # Get embeddings using DeepFace ArcFace (enforce_detection=False since already detected)
    embeddings = DeepFace.represent(face_resized, model_name="ArcFace", enforce_detection=False)

    if not embeddings:
        return None

    return embeddings[0]["embedding"]   

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

        cropped_face, face_data = crop_face_with_mtcnn(image_path)
        if cropped_face is None:
            return jsonify({"error": "No face detected!"}), 400

        embedding = get_face_embedding(cropped_face)
        if embedding is None:
            return jsonify({"error": "Failed to extract face embedding!"}), 500

        return jsonify({
            "message": "Face registered successfully",
            "face_embedding": embedding
        }), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_face", methods=["POST"])
def process_face():
    try:
        data = request.json
        image_name = data.get("image_name")
        stored_embedding = data.get("stored_embedding")

        if not image_name or not stored_embedding:
            return jsonify({"error": "Image name and stored embedding required!"}), 400

        stored_embedding = np.array(json.loads(stored_embedding))

        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image '{image_name}' not found"}), 404

        cropped_face, _ = crop_face_with_mtcnn(image_path)
        if cropped_face is None:
            return jsonify({"error": "No face detected!"}), 400

        face_embedding = get_face_embedding(cropped_face)
        if face_embedding is None:
            return jsonify({"error": "Failed to extract face embedding!"}), 500

        face_embedding = np.array(face_embedding)

        print("✅ Face Embedding Extracted:", face_embedding)
        # Ensure embeddings are valid
        if np.linalg.norm(stored_embedding) == 0 or np.linalg.norm(face_embedding) == 0:
            print("❌ Error: Invalid embeddings! Cannot compute similarity.")
            return
        # Calculate Cosine Similarity
        # Cosine similarity check
        similarity = 1 - cosine(stored_embedding, face_embedding)
        is_match = similarity > 0.65
        # Define similarity threshold

        print("\n🔍 Face Comparison Result:")
        print(f"✅ Similarity Score: {similarity:.4f}")
        print(f"✅ Match Found: {is_match}")

        return jsonify({"message": bool(is_match)}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
