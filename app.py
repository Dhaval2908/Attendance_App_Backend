import os
import json
import jwt  # PyJWT library
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
app = Flask(__name__)
load_dotenv()

# Load JWT secret and MongoDB URI from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")
MONGODB_URI = os.getenv("MONGODB_URI")

if not JWT_SECRET:
    raise ValueError("❌ JWT_SECRET environment variable is not set!")

if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI environment variable is not set!")

# Set up upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB connection setup
try:
    client = MongoClient(MONGODB_URI)
    db = client.get_database("test")  # Create/Use a database named "face_recognition"
    users_collection = db.get_collection("users")  # Collection for storing user data (face embeddings)
    # Test the connection by performing a simple query
    client.admin.command('ping')  # Ping the MongoDB server
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    raise

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

print("🔄 Loading ArcFace Model...")
print("✅ ArcFace Model Loaded Successfully!")

def verify_jwt(token):
    """ Verify JWT token and extract user data """
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return decoded
    except ExpiredSignatureError:
        return {"error": "Token expired"}
    except InvalidTokenError:
        return {"error": "Invalid token"}

def resize_image(image_path, max_size=500):
    """ Resize image to reduce processing time """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image file")

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(image_path, img)  # Save resized image

def extract_face(image_path):
    """ Detect and extract the face using DeepFace's extract_faces """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image file")

    try:
        faces = DeepFace.extract_faces(img, detector_backend="retinaface", enforce_detection=False)
        if not faces:
            return None
        return faces[0]["face"]
    except Exception as e:
        print(f"Face extraction error: {str(e)}")
        return None

def get_face_embedding(face_image):
    """ Extract face embedding using ArcFace """
    if face_image is None:
        return None

    try:
        face_resized = cv2.resize(face_image, (112, 112))
        embeddings = DeepFace.represent(face_resized, model_name="ArcFace", enforce_detection=False)
        if not embeddings:
            return None
        return embeddings[0]["embedding"]
    except Exception as e:
        print(f"❌ Error in get_face_embedding: {str(e)}")
        return None

@app.route("/register", methods=["POST"])
def register_student():
    def process_request(image_path, user_id):
        face = extract_face(image_path)
        if face is None:
            return {"error": "No face detected!"}

        embedding = get_face_embedding(face)
        if embedding is None:
            return {"error": "Failed to extract face embedding!"}

        print(f"Generated embedding: {embedding}")  # Log the generated embedding

        try:
            # Convert user_id to ObjectId if it's a valid string representation
            user_id_object = ObjectId(user_id)
        except Exception as e:
            return {"error": f"Invalid user_id format: {str(e)}"}

        # Ensure user exists in DB before updating
        existing_user = users_collection.find_one({"_id": user_id_object})
        if not existing_user:
            return {"error": "User not found in database! Please register first."}

        # Update only the embedding field
        result = users_collection.update_one(
            {"_id": user_id_object},
            {"$set": {"faceEmbedding": embedding}}  # Only update the face_embedding field
        )

        print(f"Update result: Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")

        return {
            "message": "Face registered successfully",
            "user_id": user_id,
            "face_embedding": embedding
        }

    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Authorization token is missing!"}), 401

        decoded_user = verify_jwt(token.replace("Bearer ", ""))
        if "error" in decoded_user:
            return jsonify(decoded_user), 401

        user_id = decoded_user.get("UserId")
        if not user_id:
            return jsonify({"error": "UserId is missing in JWT payload"}), 400

        image = request.files.get("image")
        if not image:
            return jsonify({"error": "Image is required!"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        resize_image(image_path)

        # Process face in a separate thread
        future = executor.submit(process_request, image_path, user_id)
        result = future.result()

        return jsonify(result), 201 if "error" not in result else 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/process_face", methods=["POST"])
def process_face():
    def compare_faces(image_path, stored_embedding):
        """ Background face comparison """
        face = extract_face(image_path)
        if face is None:
            return {"error": "No face detected!"}

        face_embedding = get_face_embedding(face)
        if face_embedding is None:
            return {"error": "Failed to extract face embedding!"}

        stored_embedding = np.array(stored_embedding)
        face_embedding = np.array(face_embedding)

        # Ensure embeddings are valid
        if np.linalg.norm(stored_embedding) == 0 or np.linalg.norm(face_embedding) == 0:
            return {"error": "Invalid embeddings! Cannot compute similarity."}

        # Cosine similarity check
        similarity = 1 - cosine(stored_embedding, face_embedding)
        is_match = similarity > 0.65

        return {"message": bool(is_match), "similarity": round(similarity, 4)}

    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Authorization token is missing!"}), 401

        decoded_user = verify_jwt(token.replace("Bearer ", ""))
        if "error" in decoded_user:
            return jsonify(decoded_user), 401

        user_id = decoded_user.get("UserId")
        if not user_id:
            return jsonify({"error": "UserId is missing in JWT payload"}), 400

        # Fetch stored face embedding from MongoDB
        user_data = users_collection.find_one({"user_id": user_id})
        if not user_data or "face_embedding" not in user_data:
            return jsonify({"error": "User face not registered"}), 404

        stored_embedding = user_data["face_embedding"]

        image = request.files.get("image")
        if not image:
            return jsonify({"error": "Image is required!"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        resize_image(image_path)

        # Process face comparison in a separate thread
        future = executor.submit(compare_faces, image_path, stored_embedding)
        result = future.result()

        return jsonify(result), 200 if "error" not in result else 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
