#!/bin/bash
cd /home/useradmin/simple_payload_autonomy_nr

# Start WFB service
sudo systemctl start wifibroadcast@drone

# Start video stream(Jeremy's command)
#rpicam-vid --camera -0 -t 0 -n --hflip --vflip --width 320 --height 240 --codec libav --libav-format mpegts -o - | gst-launch-1.0 fdsrc fd=0 ! tsdemux ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=5602

# Buffer
# sleep 30

# Run with venv python directly (no need for activate)

# Target detection Main Code
# sudo venv/bin/python target_detection.py --detect-classes 0 2 4 7 --udp-pub

