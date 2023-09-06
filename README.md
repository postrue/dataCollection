# Data Collection 

Collect data for PosTrue.

## Usage
**Single Reading** 

Writes one set of vibrations to csv, use for testing.

`python3 arduino_reading.py filename.csv`

**Session**

Creates directory inside `Data/` for each session. 

Schedules multiple readings at specified time interval for session length. Use for data collection sessions (ie. Collecting data for 60 minutes, every 10 minutes).

`python3 session.py`

