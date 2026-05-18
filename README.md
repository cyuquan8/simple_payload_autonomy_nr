# simple_payload_autonomy_nr

## Drone
1. **Install Raspberry pi OS**
    - Use the Raspberry Pi Imager to install Raspberry Pi OS 64 bit for your Raspberry Pi model.

2. **Update and Upgrade System**
    ```bash
    sudo apt update
    sudo apt upgrade
    ```

3. **Download wfb-ng**
    - Refer to [wfb-ng Setup HOWTO](https://github.com/svpcom/wfb-ng/wiki/Setup-HOWTO) and configure wfb-ng to use `wlan1`.

4. **Add Drone Key**
    - Place your drone key into `/etc`.

5. **Enable wifibroadcast@drone Service**
    ```bash
    sudo systemctl enable wifibroadcast@drone
    ```

6. **Install imx500 Dependencies**
    ```bash
    sudo apt install imx500-all
    sudo apt install imx500-tools
    ```

7. **Install OpenCV Dependencies**
    ```bash
    sudo apt install python3-opencv
    ```

8. **Clone Simple Payload Autonomy Repository**
    ```bash
    git clone -b feat/ai_camera_pub https://github.com/cyuquan8/simple_payload_autonomy_nr.git
    cd simple_payload_autonomy_nr
    ```

9. **Create Virtual Environment and Install Python Dependencies**
    ```bash
    python -m venv --system-site-packages venv
    source venv/bin/activate
    pip install -U pip
    pip install dronekit
    pip install future
    pip3 install rpi-hardware-pwm
    ```
    - *Note: For `dronekit`, modify line 2689 to use `collections.abc.mutable` if needed.*

10. **Setup Crontab for Autostart**
    ```bash
    sudo crontab -e
    chmod 775 ~/simple_payload_autonomy_nr/launcher.sh
    ```
    - Add to crontab:
      ```
      @reboot sh ~/simple_payload_autonomy_nr/launcher.sh > /home/useradmin/logs/cronlog_$(date +\%Y\%m\%d_\%H\%M\%S).log 2>&1
      ```

11. **Enable UART**
    ```bash
    sudo raspi-config
    ```
    - Go to Interface Options and enable as needed.

12. **Enable PWM & Disable Bluetooth on Raspberry Pi**
    - Add to `/boot/config.txt`:
      ```
      dtoverlay=pwm-2chan,pin=12,func=4
      dtoverlay=disable-bt
      ```

13. **Set Drone ID**
    - Add `--drone-id drone#` to the launcher.sh script.

14. **Drone WFB-ng config**
    [drone_tunnel]
    fwmark = 30  #  traffic shaper label
    ifname = 'drone-wfb'
    ifaddr = '10.5.0.2/24'
    default_route = False

    - enable listen on drone:
    [drone_mavlink]
    fwmark = 10  #  traffic shaper label
    peer = 'listen://0.0.0.0:14551'   # incoming connection for drone 1

## Ground Station

1. **Install Raspberry pi OS**
    - Use the Raspberry Pi Imager to install Raspberry Pi OS 64 bit for your Raspberry Pi model.

2. **Update and Upgrade System**
    ```bash
    sudo apt update
    sudo apt upgrade
    ```

3. **Download wfb-ng**
    - Refer to [wfb-ng Setup HOWTO](https://github.com/svpcom/wfb-ng/wiki/Setup-HOWTO) and configure wfb-ng to use `wlan1`.

4. **Add GS Key**
    - Place your GS key into `/etc`.

5. **Enable wifibroadcast@GS Service**
    ```bash
    sudo systemctl enable wifibroadcast@gs
    ```

6. **Clone Simple Payload Autonomy Repository**
    ```bash
    git clone -b feat/ai_camera_pub https://github.com/cyuquan8/simple_payload_autonomy_nr.git
    cd simple_payload_autonomy_nr
    ```
7.  **GS WFB-ng config**

    - enable connection on gs:
    [gs_mavlink]
    fwmark = 10  #  traffic shaper label
    peer = 'connect://127.0.0.1:<14551>'  # outgoing connection for drone 1

8. **Install mavproxy, console and map on GS into Venv**
    ```bash
    pip install mavproxy
    pip install wxPython
    pip install console
    pip install map
    ```
9. **GS startup commands**
    ```bash
    sudo systemctl start wifibroadcast@gs
    cd simple_payload_autonomy_nr
    source venv/bin/activate
    mavproxy.py --master=udp:127.0.0.1:14551 --console --map
    python3 simple_payload_gs.py --socketio-port 8006 --save-images
    ```
## Multi Drone Config

1. **install mavproxy on drone in the venv created previously**
    ```bash
    pip install mavproxy
    ```

2. **Add lines into launcher.sh on drone**
    ```bash
    mavproxy.py --master=/dev/ttyAMA0 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551 # The port should follow the drone id, eg. 14551 for drone 1
    ```

## Requirements
- Node.js (v14 or higher)
- npm or yarn

## Usage

Refer to the documentation for usage examples.
