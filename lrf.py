import serial.tools.list_ports
import sys
import serial
import time
import csv
from datetime import datetime
import os
import socket

# Function to parse the continuous distance frame
def parse_continuous_distance(frame: bytes) -> int:
    try:
        s = frame.decode('ascii').strip('&')
        # Remove header
        body = s[7:]  # skip '$001624'
        # Skip sequence/status (next 8 chars)
        dist_str = body[8:12]  # adjust slice if your LRF uses more digits
        distance = int(dist_str)
        return distance
    except Exception as e:
        print("Error parsing frame:", e)
        return None


UDP_IP = "172.23.144.1"
UDP_PORT = 14550   # or whatever port you configure in wfb-ng

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Serial port configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Replace with LRF port
BAUD_RATE = 115200

output_file = "lrf_data.csv"
file_exists = os.path.isfile(output_file)

# Open CSV file for appending
with open(output_file, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not file_exists:
        csvwriter.writerow(["timestamp", "distance_mm"])  # Header

    try:
        ser = serial.Serial(
            port=SERIAL_PORT,\
            baudrate=115200,\
            parity=serial.PARITY_NONE,\
            stopbits=serial.STOPBITS_ONE,\
            bytesize=serial.EIGHTBITS,\
            timeout=3)
        print("LRF logger started...")

        ser.reset_input_buffer()
        cmd=b'$00022426&c'
        ser.write(cmd)  
        #The confirmation response will be discard
        print("Continuous mode started (Ctrl+C to stop).")

        try:
            count = 0
            MAX_CONTINUOUS = 90  # send new command before hitting 100
            ser.write(b'$00022426&c')  # start continuous mode

            while True:
                data = ser.read_until(b'&')
                distance = parse_continuous_distance(data)
                if distance is not None:

                    # Log the distance with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csvwriter.writerow([timestamp, distance])
                    csvfile.flush()
                    print(f"{timestamp} - {distance} mm")

                    #Send the distance to wfb-ng
                    message = (str(distance)+'mm\n')
                    data = message.encode('utf-8')
                    sock.sendto(data, (UDP_IP, UDP_PORT))

                    count += 1
                else:
                    # Skip invalid or partial frames
                    continue

                # Re-send command periodically to avoid module's 100-frame limit
                if count >= MAX_CONTINUOUS:
                    ser.write(b'$00022426&c')
                    count = 0

        except KeyboardInterrupt:
            print("Stopping continuous measurement...")

    except KeyboardInterrupt:
        print("Stopping LRF logger...")
    finally:
        ser.close()
