import os
from datetime import datetime, date as cal
import time
import subprocess

participant = ""
session_length = ""
t_increment = ""

def session_init():
    # 1. Creates folder to dump session's csv files into 
    # 2. Creates text file with data collection session's information:
    #     - participant name
    #     - date 
    #     - how long/often data is collected

    global participant, session_length, t_increment 

    participant = input('Enter name: ')
    session_length = int(input('How long are you collecting data (mins): '))
    t_increment = int(input('How often are you collecting data (every ___ mins): '))

    date = cal.today().strftime("%m%d")
    folderpath = f"forearm/{participant}_{date}/"

    os.mkdir(folderpath)

    now = datetime.now()

    # Format the date and time
    formatted_date_time = now.strftime("%b-%d-%Y %H:%M:%S")

    f = open(f"{folderpath}session_info.txt", 'w')
    f.write(f'Participant: {participant} \nDate: {formatted_date_time} \n\n')
    f.write(f'Session Length: {session_length} minutes \nTime Increment: {t_increment} minutes \nVibrations per Reading: 30 \n\n')
    f.flush()

    return folderpath

def main():
    # Calls arduino script at each time increment during the session length.
    # Updates session's text file with session's sampling rate.
    
    folderpath = session_init()

    for i in range(0, session_length+t_increment, t_increment):
        
        os.system('afplay /System/Library/Sounds/Glass.aiff')
        print(f'----> Collecting Data, at {i} Minutes')

        filename = f'{participant}_{i}.csv'
        filepath = folderpath+filename

        sampling_rate = subprocess.check_output(['python3', 'arduino_reading.py', filepath])

        f = open(f"{folderpath}session_info.txt", 'a')
        f.write(f'{i} Minutes: {sampling_rate} \n')
        f.close()
    
        print(f'------------------------------------\n')
        
        if (i != session_length):
            time.sleep(60*t_increment)
            os.system('afplay /System/Library/Sounds/Glass.aiff')
            time.sleep(7)


if __name__ == '__main__':
    main()

