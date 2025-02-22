const jwt = require('jsonwebtoken');
const User = require('../models/User');
const Token = require('../models/token'); // Add Token model

const authMiddleware = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    console.log("aa gaya bhidu me iddhar")
    // Validate authorization header format
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'Unauthorized - Invalid authorization header format' });
    }

    const token = authHeader.split(' ')[1];
    
    // Check if token is blacklisted
    const blacklistedToken = await Token.findOne({ token });
    if (blacklistedToken) {
      return res.status(401).json({ message: 'Unauthorized - Token revoked' });
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Get user and exclude sensitive data
    const user = await User.findById(decoded.UserId)
      .select('-password -faceEmbedding')
      .lean();

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Attach user to request object
    req.user = user;
    next();

  } catch (err) {
    // Handle specific JWT errors
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ 
        message: 'Session expired',
        error: 'Token expired - Please login again'
      });
    }

    if (err.name === 'JsonWebTokenError') {
      return res.status(401).json({ 
        message: 'Invalid token',
        error: 'Malformed token - Please login again'
      });
    }

    // Handle other errors
    console.error('Authentication error:', err);
    res.status(500).json({ 
      message: 'Authentication failed',
      error: 'Internal server error' 
    });
  }
};

module.exports = authMiddleware;