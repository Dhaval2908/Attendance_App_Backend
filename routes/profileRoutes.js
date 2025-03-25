const express = require('express');
const router = express.Router();
const User = require('../models/User');
const authMiddleware = require('../middleware/authMiddleware');

// Fetch user profile
router.get('/', authMiddleware, async (req, res) => {
  try {
    res.json(req.user);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err });
  }
});

// Update profile
router.put('/update', authMiddleware, async (req, res) => {
  try {
    const { fullName, phoneNumber, country, profileImage } = req.body;
    const updatedUser = await User.findByIdAndUpdate(
      req.user._id,
      { fullName, phoneNumber, country, profileImage },
      { new: true }
    );
    res.json(updatedUser);
    console.log("profile updated")
  } catch (err) {
    res.status(500).json({ message: 'Error updating profile', error: err });
  }
});

module.exports = router;
