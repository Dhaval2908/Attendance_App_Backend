const express = require("express");
const { uploadImage, checkFace } = require("../controllers/uploadController");
const authMiddleware = require('../middleware/authmiddleware'); // Import middleware

const router = express.Router();

// ✅ Image Upload Route
router.post("/upload",authMiddleware, uploadImage);
router.get("/check-face",authMiddleware, checkFace);

module.exports = router;
