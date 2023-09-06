import sys
import serial
import datetime

def endScript(start_time, line_count, ser):
    end_time = datetime.datetime.now()
    
    ser.write(b'0') # Writes to Serial Port, ends Vibrations

    duration = end_time - start_time
    duration_in_s = duration.total_seconds() 

    print(str(line_count/duration_in_s) + " samples/second")


def main(filepath):
    # Connects to serial port and starts vibration motor.
    # Reads serial port and writes data to csv created at 'filepath'

    print_label = False
    continueReading = True
    line_count = 0

    arduino_port = "/dev/cu.usbmodem136137301" 
    baud = 9600 # bits per second to computer

    ser = serial.Serial(arduino_port, baud) 
    
    ser.write(b'1') # Writes to Serial Port, begins Vibrations

    file = open(filepath, 'a')
    
    start_time = datetime.datetime.now()

    try:
        while(continueReading):
            getData = str(ser.readline())
            data = getData[2:][:-5]

            if (data == "VIB END"):
                continueReading = False

            if ((line_count%500) == 0):
                file.write(data+", " + str(datetime.datetime.now())+"\n") 
            else:           
                file.write(data+"\n")

            line_count = line_count+1

    except KeyboardInterrupt:
        continueReading = False
        pass

    endScript(start_time, line_count, ser)


if __name__ == '__main__':
    args = sys.argv[1:]
    filepath = args[0]
    
    main(filepath)