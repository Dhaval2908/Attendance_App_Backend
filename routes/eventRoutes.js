const express = require('express');
const { getEventsForUser, getEventById } = require('../controllers/eventController');
const authMiddleware = require('../middleware/authmiddleware');

const router = express.Router();

router.get('/', authMiddleware, getEventById);
// router.get('/:id', authMiddleware, getEventById);

module.exports = router;
