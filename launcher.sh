#!/bin/bash
cd /home/useradmin/simple_payload_autonomy_nr

# Start WFB service
sudo systemctl start wifibroadcast@drone

# Buffer
sleep 30

# Run with venv python directly (no need for activate)
# Simple payload drone
sudo venv/bin/python simple_payload_drone.py --detect-classes 0 2 4 7