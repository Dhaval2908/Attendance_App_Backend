// models/Token.js
const mongoose = require('mongoose');
const { Schema } = mongoose;

const tokenSchema = new Schema({
  token: {
    type: String,
    required: true,
    index: true
  },
  expiresAt: {
    type: Date,
    required: true
  }
});

// Create TTL index to auto-delete expired tokens
tokenSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

module.exports = mongoose.models.Token || mongoose.model('Token', tokenSchema);