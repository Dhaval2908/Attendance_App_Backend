const express = require('express');
const router = express.Router();
const attendanceController = require('../controllers/attendanceController');
const adminController = require('../controllers/adminController');
const multer = require('multer');
const path = require('path');
const authMiddleware = require('../middleware/authmiddleware');  // ✅ Import auth middleware

// Multer setup
const upload = multer({ 
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, path.join(__dirname, '../uploads'));
    },
    filename: (req, file, cb) => {
      cb(null, Date.now() + path.extname(file.originalname));
    },
  })
}).single('image');

// Student routes - protect with authMiddleware
router.post('/mark-attendance', authMiddleware, upload, attendanceController.markAttendance);

// Admin routes - protect with authMiddleware (optional: add role check inside controller)
router.get('/report', authMiddleware, adminController.generateReport);
router.get('/stats', authMiddleware, attendanceController.getAttendanceStats);
router.patch('/modify-attendance', authMiddleware, adminController.modifyAttendance);

module.exports = router;
