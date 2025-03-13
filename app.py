import os
import json
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from deepface import DeepFace
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId 
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import tempfile
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)


# Load environment variables
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
MONGODB_URI = os.getenv("MONGODB_URI")

if not JWT_SECRET or not MONGODB_URI:
    raise ValueError(" Missing JWT_SECRET or MONGODB_URI environment variable!")

# Flask App Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Setup
client = MongoClient(MONGODB_URI)
db = client.get_database("test")
users_collection = db.get_collection("users")
events_collection = db.get_collection("events")
attendances_collection = db.get_collection("attendances")

# Thread pool for async tasks
executor = ThreadPoolExecutor(max_workers=4)

# Load ArcFace Model
print("🔄 Loading ArcFace Model...")
arcface_model = DeepFace.build_model("ArcFace")
print(" ArcFace Model Loaded Successfully!")

# Utility Functions
def get_object_id(value):
    try:
        return ObjectId(value)
    except Exception as e:
        print(f"Invalid ObjectId: {value} | Error: {e}")
        return None

def extract_face(image_path):
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",  
            enforce_detection=True         
        )
        
        face = faces[0]["face"] if faces else None

        if face is not None and face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8)  # 🔄 Convert float64 to uint8

        return face
    except Exception as e:
        print(f"❌ Error extracting face: {e}")
        return None


def get_face_embedding(face_image):
    if face_image is None:
        return None

    try:
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  
        face_resized = cv2.resize(face_rgb, (112, 112))  

        # ⚡ Improved: Use 'skip' to bypass DeepFace alignment
        embeddings = DeepFace.represent(
            img_path=face_resized,
            model_name="ArcFace",
            detector_backend="skip",  
            enforce_detection=False
        )
        return embeddings[0]["embedding"] if embeddings else None
    except Exception as e:
        print(f"❌ Error in get_face_embedding: {str(e)}")
        return None


def compare_faces(image_path, stored_embedding, threshold=0.28):  # 🔍 Lower threshold for stricter matching
    face = extract_face(image_path)
    if face is None:
        return {"error": "No face detected in uploaded image!"}

    uploaded_embedding = get_face_embedding(face)
    if uploaded_embedding is None:
        return {"error": "Failed to extract embedding from uploaded image!"}

    stored_embedding = np.array(stored_embedding)  # Ensure stored embedding is NumPy array
    uploaded_embedding = np.array(uploaded_embedding)

    # Normalize embeddings to improve cosine similarity precision
    stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
    uploaded_embedding = uploaded_embedding / np.linalg.norm(uploaded_embedding)

    distance = cosine(uploaded_embedding, stored_embedding)
    return {"match": distance < threshold, "distance": distance}


# Routes
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is ready"}), 200

@app.route("/register", methods=["POST"])
def register_student():
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        decoded_user = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = decoded_user.get("UserId")
        user_id_object = get_object_id(user_id)
        if not user_id_object:
            return jsonify({"error": "Invalid UserId format!"}), 400

        image = request.files.get("image")
        if not image:
            return jsonify({"error": "Image is required!"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image_path = temp_file.name
            image.save(image_path)

        try:
            face = extract_face(image_path)
            embedding = get_face_embedding(face)
        finally:
            os.remove(image_path)  #  Clean up temp file

        if not embedding:
            return jsonify({"error": "Failed to extract face embedding!"}), 400

        users_collection.update_one({"_id": user_id_object}, {"$set": {"faceEmbedding": embedding}})
        return jsonify({"message": "Face registered successfully!"}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        decoded_user = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = decoded_user.get("UserId")
        event_id = request.form.get("eventId")

        user_data = users_collection.find_one({"_id": get_object_id(user_id)})
        event_data = events_collection.find_one({"_id": get_object_id(event_id)})

        if not user_data or not event_data:
            return jsonify({"error": "User or Event not found."}), 404

        stored_embedding = np.array(user_data.get("faceEmbedding", []))
        if stored_embedding.size == 0:
            return jsonify({"error": "No face registered for this user."}), 400

        image = request.files.get("image")
        if not image:
            return jsonify({"error": "Image file is required for face verification"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image_path = temp_file.name
            image.save(image_path)

        try:
            result = compare_faces(image_path, stored_embedding)
        finally:
            os.remove(image_path)
        print(result)
        if not result.get("match"):
            return jsonify({"error": "Face verification failed!"}), 403

        attendances_collection.insert_one({
            "user": ObjectId(user_id),
            "event": ObjectId(event_id),
            "status": "present",
            "markedAt": datetime.now(),
        })

        return jsonify({"message": "Attendance marked successfully!"}), 201

    except Exception as e:
        print(f" Error marking attendance: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)