const User = require('../models/User');
const Event = require('../models/Event');
const Attendance = require('../models/Attendance');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

const haversineDistance = (coords1, coords2) => {
  const toRad = (x) => (x * Math.PI) / 180;

  const [lon1, lat1] = coords1;
  const [lon2, lat2] = coords2;

  const R = 6371000; // Earth radius in meters

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(lat1)) *
          Math.cos(toRad(lat2)) *
          Math.sin(dLon / 2) *
          Math.sin(dLon / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  const distance = R * c; // Distance in meters
  return distance;
};


exports.markAttendance = async (req, res) => {
    try {
      // console.log(req.body)
        const userId = req.user._id;
        const { eventId, latitude, longitude } = req.body;
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

        // Check if location is provided
        if (!latitude || !longitude) {
          return res.status(400).json({ error: 'Location (latitude and longitude) is required.' });
        }

        const userCoordinates = [parseFloat(longitude), parseFloat(latitude)];
        const eventCoordinates = event.location.coordinates;
        console.log("event cordinates",eventCoordinates);
        const distance = haversineDistance(userCoordinates, eventCoordinates);

        console.log(`Distance from event location: ${distance} meters`);

        const maxAllowedDistance = 20; // 100 meters threshold (adjust as needed)

        if (distance > maxAllowedDistance) {
            return res.status(403).json({
                error: 'You are too far from the event location to mark attendance.',
                distance: `${distance.toFixed(2)} meters`
            });
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
                location: req.body.location, // Placeholder if no location is provided
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
exports.getAttendanceStats = async (req, res) => {
    try {
        const userId = req.user._id; // Extract user ID from token

        const stats = await Attendance.aggregate([
            { $match: { user: userId } },
            {
                $group: {
                    _id: "$status",
                    count: { $sum: 1 }
                }
            }
        ]);

        // Convert stats to a readable format
        const attendanceStats = { present: 0, late: 0, absent: 0 };
        stats.forEach(stat => {
            attendanceStats[stat._id] = stat.count;
        });

        res.status(200).json(attendanceStats);
    } catch (error) {
        console.error('Error fetching attendance stats:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
exports.checkMultipleAttendanceStatus = async (req, res) => {
    const { eventIds } = req.body;
    const userId =req.user._id ;  // Extract user ID from token middleware
  
    if (!Array.isArray(eventIds) || eventIds.length === 0) {
      return res.status(400).json({ error: "Event IDs are required!" });
    }
  
    try {
      const attendanceRecords = await Attendance.find({
        user: userId,
        event: { $in: eventIds }
      });
  
      const statusMap = eventIds.reduce((acc, eventId) => {
        const record = attendanceRecords.find(record => record.event.toString() === eventId);
        acc[eventId] = record ? record.status : "pending";  // Default to "absent"
        return acc;
      }, {});
  
      return res.json({ statusMap });
    } catch (error) {
      console.error("❌ Error fetching attendance status:", error);
      return res.status(500).json({ error: "Internal Server Error" });
    }
  };
  