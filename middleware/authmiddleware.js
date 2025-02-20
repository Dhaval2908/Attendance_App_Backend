const jwt = require('jsonwebtoken');
const User = require('../models/User');

const authMiddleware = async (req, res, next) => {
  try {
      const token = req.headers.authorization?.split(' ')[1]; // Get token from headers
      console.log(JSON.parse(atob(token.split('.')[1])));

    if (!token) {
      return res.status(401).json({ message: 'Access Denied. No token provided.' });
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    console.log(decode)
    req.user = await User.findById(decoded.UserId).select('-password'); // Exclude password

    if (!req.user) {
      return res.status(404).json({ message: 'User not found' });
    }

    next();
  } catch (err) {
    res.status(401).json({ message: 'Invalid token', error: err });
  }
};

module.exports = authMiddleware;
