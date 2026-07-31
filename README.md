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
### Raspberry Pi 
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

### Ubuntu laptop with QGC/Mission Planner installed
1. **Rename the wfb adapter to `wlan1` using systemd `.link` or `udev`.**

2. **Check if `rtl88xxau_wfb` is the driver for the wfb adapter. Most probably it is `rtw88_8812au`.**
   ```
   ethtool -i wlan1
   ```

3. **Remove and blacklist `rtw88_8812au`:**
   ```
   sudo systemctl stop NetworkManager
   sudo systemctl stop wpa_supplicant
   sudo airmon-ng check kill
   sudo nano /etc/modprobe.d/blacklist-rtw88-8812au.conf
   ```
   Paste the following into `/etc/modprobe.d/blacklist-rtw88-8812au.conf`:
   ```
   blacklist rtw88_8812au
   blacklist rtw88_8821au
   blacklist rtw88_8822bu
   blacklist rtw88_8723du
   blacklist rtw88_usb
   blacklist rtw88_core
   ```
   Update initramfs and unload current modules
   ```
   sudo update-initramfs -u
   sudo modprobe -r rtw88_8812au rtw88_usb rtw88_core
   ```
   
4. **Install patched `RTL8812AU` driver:**
   ```
   sudo apt-get install dkms
   git clone -b v5.2.20 https://github.com/svpcom/rtl8812au.git
   cd rtl8812au/
   sudo ./dkms-install.sh
   ```
   Load new driver
   ```
   sudo modprobe 88XXau_wfb
   ```
   Verify:
   ```
   ethtool -i wlan1
   ```
   Refer to [wfb-ng Setup HOWTO](https://github.com/svpcom/wfb-ng/wiki/Setup-HOWTO) for config file setup, notably `/etc/sysctl.conf`, `/etc/wifibroadcast.cfg`, `/etc/default/wifibroadcast`, `/etc/NetworkManager/NetworkManager.conf`, `/etc/dhcpcd.conf`.

6. **Add GS Key**
    - Place your GS key into `/etc`.

7. **Enable and start wifibroadcast@gs service**
    ```bash
    sudo systemctl enable wifibroadcast@gs
    sudo systemctl start wifibroadcast@gs
    ```
    
9. **Install QGroundControl or Mission Planner**

   On the GUI, choose UDP connection and enter the port number matching the drone

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
