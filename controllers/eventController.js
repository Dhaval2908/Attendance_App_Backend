const Event = require('../models/Event');
const User = require('../models/User')

// Get only events where the user is registered
const getEventsForUser = async (req, res) => {
    try {
        const userId = req.user.id;
        if (!userId) return res.status(400).json({ message: 'User ID not found' });

        const events = await Event.find({ registeredStudents: userId }).populate('registeredStudents', 'fullName email');
        res.json(events.length > 0 ? events : { message: 'No upcoming events' });
    } catch (error) {
        res.status(500).json({ message: 'Server error', error });
    }
};

// Get a single event by ID (Ensures user is registered for it)
const getEventById = async (req, res) => {
    try {
        const userId = req.user._id; // Middleware se user ID mil rahi hai

        // Pehle sirf event IDs le lo user document se
        const user = await User.findById(userId).select('registeredEvents'); 

        console.log(user)
        if (!user || user.registeredEvents.length === 0) {
            return res.status(404).json({ message: 'No registered events found' });
        }

        // Ab event IDs ka data fetch karo
        const events = await Event.find({ _id: { $in: user.registeredEvents } })
            .populate('creator', 'fullName email'); // Event ke creator ka data bhi bhejna hai

        res.json(events);
    } catch (error) {
        console.error('Error fetching registered events:', error);
        res.status(500).json({ message: 'Server error', error });
    }
};


module.exports = { getEventsForUser, getEventById };
