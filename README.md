# Data Collection 

Collect data for PosTrue.

## Usage
1. Upload `reading_arduino.ino` to board
2. Confirm vibration motor can be turned on/off by inputting `1` (VIB ON) and `0` (VIB OFF) into serial monitor
3. Update arduino port in `arduino_reading.py` based on your computer's serial port information 

**Single Reading**

Writes one set of vibrations to csv, use for testing.

`python3 arduino_reading.py filename.csv`

**Session**

Creates directory inside `Data/` for each session. 

Schedules multiple readings at specified time interval for session length. Use for data collection sessions (ie. Collecting data for 60 minutes, every 10 minutes).

`python3 session.py`


