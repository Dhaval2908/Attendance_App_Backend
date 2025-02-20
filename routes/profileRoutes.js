const express = require('express');
const router = express.Router();
const User = require('../models/User');
const authMiddleware = require('../middleware/authmiddleware'); // Import middleware

// GET: Fetch user profile (No need to send userId in URL)
router.get('/', authMiddleware, async (req, res) => {
  try {
    console.log(req.user)
    res.json(req.user); // User details from middleware
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err });
  }
});

// POST: Update user profile (Only user can update their own profile)
router.post('/', authMiddleware, async (req, res) => {
  try {
    const updatedUser = await User.findByIdAndUpdate(
      req.user._id,
      { $set: req.body },
      { new: true }
    );
    res.json(updatedUser);
  } catch (err) {
    res.status(500).json({ message: 'Server error', error: err });
  }
});

module.exports = router;
