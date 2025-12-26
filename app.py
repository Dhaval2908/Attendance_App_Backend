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
from mtcnn import MTCNN
import time
from datetime import datetime
import pytz

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
detector = MTCNN(min_face_size=40)
# Utility Functions
def get_object_id(value):
    try:
        return ObjectId(value)
    except Exception as e:
        print(f"Invalid ObjectId: {value} | Error: {e}")
        return None

def extract_face(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(" Failed to read image")
            return None

        faces = detector.detect_faces(img)
        if not faces:
            print(" No face detected")
            return None

        x, y, w, h = faces[0]["box"]
        face = img[y:y+h, x:x+w]

        # Resize directly to model's required size
        face_resized = cv2.resize(face, (112, 112))

        if face_resized.dtype != np.uint8:
            face_resized = (face_resized * 255).astype(np.uint8)

        return face_resized
    except Exception as e:
        print(f" Error extracting face: {e}")
        return None

def get_face_embedding(face_image):
    if face_image is None:
        return None

    try:
        embeddings = DeepFace.represent(
            img_path=face_image,
            model_name="ArcFace",
            detector_backend="skip",  #  Skip redundant detection
            enforce_detection=False
        )

        return embeddings[0]["embedding"] if embeddings else None
    except Exception as e:
        print(f" Error in get_face_embedding: {str(e)}")
        return None


def haversine_distance(coord1, coord2):
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    R = 6371  # Radius of Earth in kilometers
    return R * c * 1000  # Return distance in meters

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def compare_faces(image_path, stored_embedding, threshold=0.28):  # 🔍 Lower threshold for stricter matching
    face = extract_face(image_path)
    if face is None:
        return {"error": "No face detected. Please try again!"}

    uploaded_embedding = get_face_embedding(face)
    if uploaded_embedding is None:
        return {"error": "Failed to extract embedding from uploaded image!"}

    stored_embedding = np.array(stored_embedding)  # Ensure stored embedding is NumPy array
    uploaded_embedding = np.array(uploaded_embedding)

    distance = cosine(uploaded_embedding, stored_embedding)
    print("distance:", distance)
    return {"match": distance < threshold, "distance": distance}


# Routes
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is ready"}), 200

@app.route("/register", methods=["POST"])
def register_student():
    start_time = time.time()
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
        end_time = time.time()  #  End time tracking
        execution_time = round(end_time - start_time, 2)  #  Calculate time in seconds
        print(f"🕒 /register API executed in {execution_time} seconds")
        return jsonify({"message": "Face registered successfully!"}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    start_time = time.time()
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        decoded_user = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = decoded_user.get("UserId")
        event_id = request.form.get("eventId")

        user_data = users_collection.find_one({"_id": get_object_id(user_id)})
        event_data = events_collection.find_one({"_id": get_object_id(event_id)})
        print(user_data)
        print(event_data)

        if not user_data or not event_data:
            return jsonify({
                "error": "User or Event not found.",
            }), 404  # Not Found

        stored_embedding = np.array(user_data.get("faceEmbedding", []))
        if stored_embedding.size == 0:
            return jsonify({
                "error": "Please register your face first.",
            }), 400  # Bad Request


        #  Location check
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        if not latitude or not longitude:
            return jsonify({
                "error": "Please provide location data.",
            }), 400  # Bad Request

        user_coordinates = [float(longitude), float(latitude)]
        event_coordinates = event_data["location"]["coordinates"]
        distance = haversine_distance(user_coordinates, event_coordinates)

        max_allowed_distance = 20  # 20 meters threshold (adjust as needed)
        if distance > max_allowed_distance:
            return jsonify({
                "error": "You are too far from the event location.",
            }), 403  # Forbidden
        
        image = request.files.get("image")
        if not image:
            return jsonify({
                "error": "Please capture an image for face verification."
            }), 400  # Bad Request


        # with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        #     image_path = temp_file.name
        #     image.save(image_path)
         #  Save uploaded image in 'Uploads' directory
        upload_folder = os.path.join(os.path.dirname(__file__), "Uploads")
        os.makedirs(upload_folder, exist_ok=True)  # Ensure 'Uploads' directory exists
        image_filename = f"{user_id}_{event_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = os.path.join(upload_folder, image_filename)
        image.save(image_path)
        
        try:
            result = compare_faces(image_path, stored_embedding)
        finally:
            os.remove(image_path)
        
        print(result)
        if not result.get("match"):
            return jsonify({
                "error": "Face does not match with registered face."
            }), 403  # Forbidden
        
        #  Check event timing and attendance status
        # Timezone conversion to America/Toronto
        toronto_tz = pytz.timezone("America/Toronto")
        
        current_time = datetime.now()
        current_time = datetime.now(toronto_tz)

        event_start_time = event_data["startTime"]
        event_start_time = event_data["startTime"].replace(tzinfo=pytz.utc).astimezone(toronto_tz)

        event_end_time = event_data["endTime"]
        event_end_time = event_data["endTime"].replace(tzinfo=pytz.utc).astimezone(toronto_tz)
        
        buffer_minutes = event_data["bufferMinutes"]

        attendance_status = "present"
        
        print("event_start_time:",event_start_time)
        print("current_time:",current_time)

        late_minutes = 0
        
        if current_time > event_start_time:
            late_minutes = (current_time - event_start_time).seconds // 60
            if late_minutes < buffer_minutes:
                attendance_status = "present"
            else:
                attendance_status = "late"
           

        if current_time > event_end_time:
            attendance_status = "absent"
            late_minutes = 0  # No late minutes if absent
        
        # Extract location from request body
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        location = {"latitude": latitude, "longitude": longitude}


        #  Save attendance record
        attendances_collection.insert_one({
            "user": ObjectId(user_id),
            "event": ObjectId(event_id),
            "status": attendance_status,
            "lateMinutes": late_minutes,
            "markedAt": current_time,
            "location": location,  # Add location to attendance record
            "modifiedBy": ObjectId(user_id),  # User who modified the attendance
        })
        end_time = time.time()  #  End time tracking
        execution_time = round(end_time - start_time, 2)  #  Calculate time in seconds
        print(f"🕒 /register API executed in {execution_time} seconds")
        return jsonify({
            "message": "Attendance marked successfully!",
            "status": "success"
        }), 201  # Created

    except jwt.ExpiredSignatureError:
        return jsonify({
            "error": "Token expired.",
            "message": "Your session has expired. Please log in again."
        }), 401  # Unauthorized
    
    except jwt.InvalidTokenError:
        return jsonify({
            "error": "Invalid token.",
            "message": "Your token is invalid. Please log in again."
        }), 401  # Unauthorized

    except Exception as e:
        print(f" Error marking attendance: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error.",
            "message": "An unexpected error occurred. Please try again later."
        }), 500  # Internal Server Error


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, threaded=True,debug=True)

