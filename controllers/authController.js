const User = require('../models/User');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

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
