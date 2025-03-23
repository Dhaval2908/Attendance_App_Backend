const mongoose = require('mongoose');
const { Schema } = mongoose;
const bcrypt = require('bcrypt');

const userSchema = new Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    match: /^[a-zA-Z0-9._-]+@(uwindsor\.ca)$/i
  },
  studentId: {
    type: String,
    required: true,
    unique: true
  },
  fullName: {
    type: String,
    required: true
  },
  phoneNumber: {
    type: String,
    default: ''
  },
  country: {
    type: String,
    default: ''
  },
  profileImage: {
    type: String, // Store image URL
    default: ''
  },
  password: {
    type: String,
    required: true
  },
  faceEmbedding: {
    type: [Number],
    required: true
  },
  role: {
    type: String,
    enum: ['admin', 'student'],
    default: 'student'
  },
  registeredEvents: [{
    type: Schema.Types.ObjectId,
    ref: 'Event'
  }]
}, { timestamps: true });

userSchema.methods.comparePassword = async function (candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.models.User || mongoose.model('User', userSchema);