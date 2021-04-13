import serial
import time
import datetime
import json
import redis

data = {}
data['colocation_data'] = []

DWM = serial.Serial(port="/dev/tty.usbmodem0007601250411", baudrate=115200)
print("Connected to " + DWM.name)
DWM.write("\r\r".encode())
print("Encode")
time.sleep(1)
DWM.write("lec\r".encode())
time.sleep(1)

while True:
    try:
        line=DWM.readline()
        if(line):
            if len(line)>=15:
                # print(line)
                parse=line.decode().split(",")
                # pos_AN0=(parse[parse.index("AN0")+2],parse[parse.index("AN0")+3],parse[parse.index("AN0")+4])
                dist_AN0=parse[parse.index("AN0")+5]
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                # Add to JSON
                data['co-location_data'].append({
                'timestamp': timestamp,
                'distance': dist_AN0
                })
                print(timestamp, dist_AN0)
            else:
                print("Distance not calculated: ",line.decode())

            with open('Big_Experiment.txt', 'w') as outfile:
                json.dump(data, outfile)
    
    except Exception as ex:
        print(ex)
        break

DWM.write("\r".encode())
DWM.close()
