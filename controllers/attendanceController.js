const User = require('../models/User');
const Event = require('../models/Event');
const Attendance = require('../models/Attendance');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

exports.markAttendance = async (req, res) => {
    try {
      // console.log(req.body)
        const userId = req.user._id;
        const { eventId } = req.body;
        // console.log(req.body)

        const user = await User.findById(userId);
        // console.log(user)
        const event = await Event.findById(eventId);
        // console.log(event)
        
        if (!user || !event) {
            return res.status(404).json({ error: 'User or Event not found' });
        }

          // ✅ Check if user is registered for the event
          if (!event.registeredStudents.includes(userId)) {
              return res.status(403).json({ error: 'User not registered for this event' });
          }

        // ✅ Step 1: Upload image to Flask for face embedding extraction
        if (!req.file) {
            return res.status(400).json({ error: 'Image file is required for face verification' });
        }
        
        // Get the uploaded image filename
        const imageName = req.file.filename;
        // console.log("Uploaded Image Name:", imageName);

        const flaskApiUrl = process.env.FLASK_API_URL + "process_face";

        const formData = new FormData();
        const flaskResponse = await axios.post(flaskApiUrl, {
                image_name: imageName,
                stored_embedding: JSON.stringify(user.faceEmbedding),
        });
        let isFaceVerified = false;
      
        if (flaskResponse.status === 200 && flaskResponse.data.message) {
          console.log('Face verification passed.');
          isFaceVerified = true;
          console.log(flaskResponse.message)
        }else{
          console.error('Unexpected response from Flask:', flaskResponse.status);
          return res.status(500).json({ error: 'Face verification failed.' });
        }

        
        if (isFaceVerified){
          // ✅ Check event timing
            const currentTime = new Date();
            const eventStartTime = new Date(event.startTime);
            const eventEndTime = new Date(event.endTime);

            let attendanceStatus = 'present';
            let lateMinutes = 0;

            // If user arrives after event start time, mark them as late
            if (currentTime > eventStartTime) {
                attendanceStatus = 'late';
                // Calculate late minutes
                lateMinutes = Math.floor((currentTime - eventStartTime) / (1000 * 60)); // in minutes
            }

            // If user arrives after the event end time, mark them as absent
            if (currentTime > eventEndTime) {
                attendanceStatus = 'absent';
                lateMinutes = 0;  // No late minutes if absent
            }

            // Create the attendance entry
            const attendanceRecord = new Attendance({
                user: userId,
                event: eventId,
                status: attendanceStatus,
                lateMinutes: lateMinutes,
                markedAt: currentTime,
                location: req.body.location || { type: 'Point', coordinates: [0, 0] }, // Placeholder if no location is provided
                modifiedBy: userId,
            });

            // Save attendance record
            await attendanceRecord.save();

            return res.status(201).json({
                message: `Attendance marked as ${attendanceStatus}`,
                userId,
                eventId,
                attendanceStatus,
                lateMinutes
            });
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        res.status(500).json({ error: error.message });
    }
};