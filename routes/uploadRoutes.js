const express = require("express");
const { uploadImage } = require("../controllers/uploadController");

const router = express.Router();

// ✅ Image Upload Route
router.post("/upload", uploadImage);

module.exports = router;
