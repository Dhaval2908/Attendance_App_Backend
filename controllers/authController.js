const User = require('../models/User');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');


// Add to existing imports
exports.logout = async (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify and decode the token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Add token to blacklist
    await Token.create({
      token,
      expiresAt: new Date(decoded.exp * 1000) // Convert JWT exp to Date
    });

    res.status(200).json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token already expired' });
    }
    
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ error: 'Invalid token' });
    }

    res.status(500).json({ error: 'Internal server error' });
  }
};



// Signup API
exports.signup = async (req, res) => {
  try {
    const { email, studentId, fullName, password, role } = req.body;

    // Check if User already exists
    const existingUser = await User.findOne({ $or: [{ email }, { studentId }] });
    if (existingUser) {
      return res.status(400).json({ error: 'Email or Student ID already exists' });
    }

    // Hash the password before saving
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);
    
    // Create new User

    const user = await User.create({
      email,
      studentId,
      fullName,
      password: hashedPassword, // Store hashed password
      role: role || 'student'
    })
 

    await user.save();

    // Generate JWT token
    const token = jwt.sign(
      { UserId: user._id, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '1h' }
    );

    res.status(201).json({ token, User: { id: user._id, email: user.email, role: user.role } });
  } catch (error) {
  
    res.status(500).json({ error: 'Internal server error' });
  }
};

// Login API
exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Find User by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'User Does not exist' });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, user.password);
    
    if (!isMatch) {
      return res.status(401).json({ error: 'Password Does not Match' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { UserId: user._id, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '1h' }
    );

    res.json({ token, User: { id: user._id, email: user.email, role: user.role } });
  } catch (error) {
    
    res.status(500).json({ error: 'Internal server error' });
  }
};
