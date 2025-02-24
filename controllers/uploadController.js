// const fs = require("fs");
// const multer = require("multer");
// const path = require("path");

// // ✅ Ensure uploads folder exists
// const uploadDir = path.join(__dirname, "../uploads");
// if (!fs.existsSync(uploadDir)) {
//   fs.mkdirSync(uploadDir, { recursive: true });
// }

// // ✅ Multer Storage Config
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => {
//     cb(null, uploadDir);  // Use absolute path to avoid issues
//   },
//   filename: (req, file, cb) => {
//     cb(null, Date.now() + path.extname(file.originalname));
//   },
// });

// const upload = multer({ storage: storage }).single("image");

// // ✅ Handle Image Upload
// const uploadImage = (req, res) => {
//   upload(req, res, (err) => {
//     if (err) {
//       return res.status(500).json({ error: "Upload failed", details: err.message });
//     }
//     if (!req.file) {
//       return res.status(400).json({ error: "No file uploaded" });
//     }
//     res.json({ message: "✅ Image uploaded successfully", filename: req.file.filename });
//   });
// };

// module.exports = { uploadImage };
const fs = require("fs");
const multer = require("multer");
const path = require("path");
const axios = require("axios");
const User = require("../models/User"); // Import User model
require('dotenv').config();

// ✅ Ensure uploads folder exists
const uploadDir = path.join(__dirname, "../uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ✅ Multer Storage Config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage }).single("image");

// ✅ Handle Image Upload and Call Flask API
const uploadImage = async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(500).json({ error: "Upload failed", details: err.message });
    }
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    try {
      // ✅ Get User ID from Middleware
      const userId = req.user._id;// Middleware should attach user info to `req.user`
      if (!userId) {
        return res.status(401).json({ error: "Unauthorized: User ID not found" });
      }

      const flaskApiUrl = process.env.FLASK_API_URL + "register";

      const flaskResponse = await axios.post(
        flaskApiUrl,
        { image: fs.createReadStream(req.file.path) },
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      console.log(flaskResponse)
      if (!flaskResponse.data.face_embedding) {
        return res.status(400).json({ error: "Face embedding not received" });
      }

      const faceEmbedding = flaskResponse.data.face_embedding;

      // ✅ Store Face Embedding in User Model
      const user = await User.findById(userId);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      user.faceEmbedding = faceEmbedding;
      await user.save();

      res.json({
        message: "✅ Image uploaded & face embedding stored",
        filename: req.file.filename,
        faceEmbedding,
      });
    } catch (error) {
      console.error("Error processing image:", error);
      res.status(500).json({ error: "Failed to process image", details: error.message });
    }
  });
};

const checkFace = async (req, res) => {
  try {
    const user = await User.findById(req.user._id);

    if (user && user.faceEmbedding) {
      return res.json({ registered: true });
    } else {
      return res.json({ registered: false });
    }
  } catch (error) {
    console.error("Error checking face registration:", error);
    res.status(500).json({ error: "Server error" });
  }
};
module.exports = { uploadImage,checkFace };
