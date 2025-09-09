import argparse
import base64
import json
import logging
import os
import piexif
import socket
import threading
import time

from datetime import datetime
from message_types import (
    MessageType,
    LandStatus,
    RTLStatus, 
    TakeoffStatus, 
    WaypointStatus, 
)
from typing import Any

class SimplePayloadGroundStation:
    def __init__(self, args: argparse.Namespace):
        
        # Logger
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging._nameToLevel.get(args.log_level, "DEBUG"))
        # Formatter
        formatter = logging.Formatter(
            "%(name)s - %(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        # Add StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        # Add FileHandler with timestamp
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ground_station_{timestamp}.log"
        log_filepath = os.path.join(args.log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging._nameToLevel.get(args.log_level, "DEBUG"))
        self._logger.addHandler(file_handler)
        
        # Parameters
        self.image_save_dir = args.image_save_dir
        self.save_images = args.save_images
        self.socket_timeout = args.socket_timeout
        self.udp_buffer_size = args.udp_buffer_size
        self.udp_ip = args.udp_ip
        self.udp_port = args.udp_port
        
        # Threads
        self._udp_receiver_thread = None
        self._udp_receiver_active = False
        # Shutdown event for graceful termination
        self._shutdown_event = threading.Event()
        
        # Setup UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.udp_ip, self.udp_port))
        
        # Statistics
        self._messages_received = 0
        self._last_message_time = None
        
        # Create timestamped image save directory if needed
        if self.save_images:
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.image_save_dir = os.path.join(
                self.image_save_dir, 
                f"gs_session_{session_timestamp}"
            )
            os.makedirs(self.image_save_dir, exist_ok=True)
    
    #########################
    ### Getters / Setters ###
    #########################

    @property
    def logger(self) -> logging.Logger:
        """
        Get logger instance.
        
        Returns:
            logging.Logger: Logger instance for this class
        """
        return self._logger
    
    ######################################
    ### Message queue helper functions ###
    ######################################
    
    @staticmethod
    def _decimal_to_dms(decimal_degrees: float) -> list[tuple[int, int]]:
        """
        Convert decimal degrees to DMS format for EXIF GPS data.
        
        Args:
            decimal_degrees: Coordinate in decimal degrees
            
        Returns:
            list[tuple[int, int]]: DMS in EXIF format 
                [(degrees, 1), (minutes, 1), (seconds_num, seconds_den)]
        """
        abs_degrees = abs(decimal_degrees)
        degrees = int(abs_degrees)
        minutes_float = (abs_degrees - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        
        # EXIF format: 
        # [(degrees, 1), (minutes, 1), (seconds_numerator, seconds_denominator)]
        return [(degrees, 1), (minutes, 1), (int(seconds * 1000), 1000)]
    
    def _decode_message(self, data: bytes) -> dict[str, Any] | None:
        """
        Decode received UDP message from drone.
        
        Args:
            data: Raw UDP message bytes
            
        Returns:
            dict: Decoded message data
        """
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            # self._logger.debug(f"Failed to decode JSON message: {e}")
            return None
        except UnicodeDecodeError as e:
            # self._logger.debug(f"Failed to decode UTF-8 data: {e}")
            return None
        
    def _log_message_stats(self) -> None:
        """
        Log message reception statistics.
        
        Args:
            None
            
        Returns:
            None
        """
        current_time = time.time()
        if self._last_message_time is not None:
            interval = current_time - self._last_message_time
            rate = 1.0 / interval if interval > 0 else 0
            self._logger.debug(
                f"Message rate: {rate:.1f} Hz, Total: {self._messages_received}"
            )
        self._last_message_time = current_time
    
    def _process_detection_message(self, message: dict) -> None:
        """
        Process a detection message from the drone.
        
        Args:
            message: Decoded detection message dictionary
            
        Returns:
            None
        """
        # Extract message components
        detections_data = message.get('detections', [])
        image_data = message.get('image')
        location_data = message.get(
            'location', 
            {'lat': None, 'lon': None, 'alt': None}
        )
        timestamp = message.get('timestamp', 'unknown')
        drone_id = message.get('id', 'unknown')
        
        # Log detection information
        for detection in detections_data:
            label = detection['label']
            confidence = detection['confidence']
            vehicle_lat = location_data['lat']
            vehicle_lon = location_data['lon'] 
            vehicle_alt = location_data['alt']
            pred_loc = detection['pred_loc']
            
            self._logger.info(
                f"[ID: {drone_id}] {label} detected with {confidence:.2f} "
                f"confidence at vehicle pos lat:{vehicle_lat}, "
                f"lon:{vehicle_lon}, alt:{vehicle_alt} -> predicted object "
                f"pos lat:{pred_loc['lat']}, lon:{pred_loc['lon']}, "
                f"bearing:{pred_loc['bearing_deg']}, "
                f"distance:{pred_loc['distance_m']} (time: {timestamp})"
            )
        
        # Save image if requested
        if image_data:
            if self.save_images:
                saved_path = self._save_image(
                    image_data, 
                    timestamp, 
                    location_data
                )
                if saved_path:
                    self._logger.debug(f"Saved image to: {saved_path}")
        else:
            self._logger.debug("No image data in message")
    
    def _process_land_message(self, message: dict) -> None:
        """
        Process a land message from the drone.
        
        Args:
            message: Decoded land message dictionary
            
        Returns:
            None
        """
        # Extract message components
        drone_id = message.get('id', 'unknown')
        status = message.get('status', 'unknown')
        alt = message.get('altitude', 'unknown')
        timestamp = message.get('timestamp', 'unknown')
        # Log land information based on status
        if status == LandStatus.STARTED:
            self._logger.info(
                f"[ID: {drone_id}] Land STARTED - Current altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == LandStatus.COMPLETED:
            self._logger.info(
                f"[ID: {drone_id}] Land COMPLETED - Landing altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == LandStatus.ABORTED:
            self._logger.warning(
                f"[ID: {drone_id}] Land ABORTED - Altitude: {alt}m "
                f"(time: {timestamp})"
            )
        else:
            self._logger.warning(
                f"[ID: {drone_id}] Unknown land status: {status} "
                f"- Altitude: {alt}m (time: {timestamp})"
            )

    def _process_rtl_message(self, message: dict) -> None:
        """
        Process an RTL message from the drone.
        
        Args:
            message: Decoded RTL message dictionary
            
        Returns:
            None
        """
        # Extract message components
        drone_id = message.get('id', 'unknown')
        status = message.get('status', 'unknown')
        alt = message.get('altitude', 'unknown')
        timestamp = message.get('timestamp', 'unknown')
        # Log RTL information based on status
        if status == RTLStatus.STARTED:
            self._logger.info(
                f"[ID: {drone_id}] RTL STARTED - Current altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == RTLStatus.COMPLETED:
            self._logger.info(
                f"[ID: {drone_id}] RTL COMPLETED - Landing altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == RTLStatus.ABORTED:
            self._logger.warning(
                f"[ID: {drone_id}] RTL ABORTED - Altitude: {alt}m "
                f"(time: {timestamp})"
            )
        else:
            self._logger.warning(
                f"[ID: {drone_id}] Unknown RTL status: {status} "
                f"- Altitude: {alt}m (time: {timestamp})"
            )

    def _process_takeoff_message(self, message: dict) -> None:
        """
        Process a takeoff message from the drone.
        
        Args:
            message: Decoded takeoff message dictionary
            
        Returns:
            None
        """
        # Extract message components
        drone_id = message.get('id', 'unknown')
        status = message.get('status', 'unknown')
        alt = message.get('altitude', 'unknown')
        timestamp = message.get('timestamp', 'unknown')
        # Log takeoff information based on status
        if status == TakeoffStatus.STARTED:
            self._logger.info(
                f"[ID: {drone_id}] Takeoff STARTED - Target altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == TakeoffStatus.COMPLETED:
            self._logger.info(
                f"[ID: {drone_id}] Takeoff COMPLETED - Final altitude: {alt}m "
                f"(time: {timestamp})"
            )
        elif status == TakeoffStatus.ABORTED:
            self._logger.warning(
                f"[ID: {drone_id}] Takeoff ABORTED - Altitude: {alt}m "
                f"(time: {timestamp})"
            )
        else:
            self._logger.warning(
                f"[ID: {drone_id}] Unknown takeoff status: {alt} "
                f"- Altitude: {alt}m (time: {timestamp})"
            )

    def _process_telemetry_message(self, message: dict) -> None:
        """
        Process a telemetry message from the drone.
        
        Args:
            message: Decoded telemetry message dictionary
            
        Returns:
            None
        """
        # Log telemetry information
        drone_id = message.get('id', 'unknown')
        self._logger.info(f"[ID: {drone_id}] Telemetry message: {message}")
    
    def _process_waypoint_message(self, message: dict) -> None:
        """
        Process a waypoint message from the drone.
        
        Args:
            message: Decoded waypoint message dictionary
            
        Returns:
            None
        """
        # Extract message components
        drone_id = message.get('id', 'unknown')
        status = message.get('status', 'unknown')
        waypoint_index = message.get('waypoint_index', 'unknown')
        total_waypoints = message.get('total_waypoints', 'unknown')
        lat = message.get('lat', 'unknown')
        lon = message.get('lon', 'unknown')
        timestamp = message.get('timestamp', 'unknown')
        
        # Log waypoint information based on status
        if status == WaypointStatus.STARTED:
            self._logger.info(
                f"[ID: {drone_id}] Waypoint {waypoint_index}/{total_waypoints} "
                f"STARTED - Target: ({lat}, {lon}) (time: {timestamp})"
            )
        elif status == WaypointStatus.COMPLETED:
            self._logger.info(
                f"[ID: {drone_id}] Waypoint {waypoint_index}/{total_waypoints} "
                f"COMPLETED - Location: ({lat}, {lon}) (time: {timestamp})"
            )
        elif status == WaypointStatus.ABORTED:
            self._logger.warning(
                f"[ID: {drone_id}] Waypoint {waypoint_index}/{total_waypoints} "
                f"ABORTED - Location: ({lat}, {lon}) (time: {timestamp})"
            )
        else:
            self._logger.warning(
                f"[ID: {drone_id}] Unknown waypoint status: {status} "
                f"- Waypoint {waypoint_index}/{total_waypoints} at "
                f"({lat}, {lon}) (time: {timestamp})"
            )

    def _save_image(self, 
            image_data: str, 
            timestamp: str, 
            location_data: dict | None = None,
        ) -> str | None:
        """
        Save base64 encoded image to file with GPS metadata.
        
        Args:
            image_data: Base64 encoded image string
            timestamp: Timestamp string for filename
            location_data: Dictionary with lat/lon/alt data
            
        Returns:
            str: Saved image filepath
        """
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data.encode('utf-8'))
            
            # Create filename
            clean_timestamp = timestamp.replace(':', '-').replace('.', '_')
            filename = f"{clean_timestamp}.jpg"
            filepath = os.path.join(self.image_save_dir, filename)

            # Save image first
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            
            # Add GPS EXIF metadata if location data is available
            if location_data and \
            all(location_data.get(k) is not None for k in ['lat', 'lon', 'alt']):
                lat = float(location_data['lat'])
                lon = float(location_data['lon']) 
                alt = float(location_data['alt'])
                # Create GPS EXIF data
                gps_dict = {
                    piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
                    piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
                    piexif.GPSIFD.GPSLatitude: self._decimal_to_dms(lat),
                    piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
                    piexif.GPSIFD.GPSLongitude: self._decimal_to_dms(lon),
                    piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1, # 0=above, 1=below sea level
                    piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100), # Altitude in 2dp
                }
                exif_dict = {"GPS": gps_dict}
                exif_bytes = piexif.dump(exif_dict)
                # Insert EXIF into saved file
                piexif.insert(exif_bytes, filepath)
                self._logger.debug(
                    f"Added GPS metadata: lat={lat}, lon={lon}, alt={alt}"
                )

            return filepath
        except Exception as e:
            self._logger.error(f"Failed to save image: {e}")
            return None
    
    ################################
    ### Thread handler functions ###
    ################################
    
    def _cleanup_resources(self) -> None:
        """
        Private helper to cleanup ground station resources and stop 
        background threads.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Cleaning up ground station resources...")
        if self._udp_receiver_thread is not None:
            self._stop_udp_receiver()
        if hasattr(self, '_sock') and self._sock:
            self._sock.close()
            self._logger.info("Closed UDP socket")
        self._shutdown_event.clear()
        self._logger.info("Ground station cleanup complete")

    def _start_udp_receiver(self) -> None:
        """
        Start the UDP receiver thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If UDP receiver is already started
        """
        if self._udp_receiver_thread is not None:
            raise RuntimeError("UDP receiver already started")
        
        self._udp_receiver_active = True
        self._udp_receiver_thread = threading.Thread(
            target=self._udp_receiver_worker,
            daemon=True,
            name="UDP-Receiver"
        )
        self._udp_receiver_thread.start()
        self._logger.info("Started UDP receiver thread")
    
    def _stop_udp_receiver(self) -> None:
        """
        Stop the UDP receiver thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._udp_receiver_thread is not None:
            self._logger.info("Stopping UDP receiver...")
            self._udp_receiver_active = False
            self._udp_receiver_thread.join(timeout=5.0)
            if self._udp_receiver_thread.is_alive():
                self._logger.warning(
                    "UDP receiver thread did not stop gracefully"
                )
            else:
                self._logger.info("UDP receiver thread stopped")
            self._udp_receiver_thread = None
    
    ###############################
    ### Thread worker functions ###
    ###############################
    
    def _udp_receiver_worker(self) -> None:
        """
        Worker thread function that receives and processes UDP messages.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("UDP receiver worker started")
        self._logger.info(f"Listening on {self.udp_ip}:{self.udp_port}")
        
        while self._udp_receiver_active and not self._shutdown_event.is_set():
            try:
                # Receive UDP message
                data, addr = self._sock.recvfrom(self.udp_buffer_size)
                self._messages_received += 1
                
                self._logger.debug(
                    f"Received message from {addr}, size: {len(data)} bytes"
                )
                self._log_message_stats()
                
                # Decode message
                message = self._decode_message(data)
                if message is None:
                    continue
                
                # Handle message based on type
                message_type = message.get('type', MessageType.UNKNOWN)
                if message_type == MessageType.DETECTION:
                    self._process_detection_message(message)
                elif message_type == MessageType.TELEMETRY:
                    self._process_telemetry_message(message)
                elif message_type == MessageType.TAKEOFF:
                    self._process_takeoff_message(message)
                elif message_type == MessageType.RTL:
                    self._process_rtl_message(message)
                elif message_type == MessageType.WAYPOINT:
                    self._process_waypoint_message(message)
                elif message_type == MessageType.LAND:
                    self._process_land_message(message)
                else:
                    self._logger.warning(
                        f"Unknown message type: {message_type}"
                    )
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                self._logger.error(f"UDP receiver error: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        self._logger.info("UDP receiver worker stopped")
    
    #########################
    ### Exposed functions ###
    #########################
    
    def run(self) -> None:
        """
        Run the ground station. This method blocks until interrupted.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info(
            "Running ground station - listening for drone messages"
        )
        
        # Start UDP receiver
        self._start_udp_receiver()
        
        # Wait for shutdown event while receiver processes messages
        self._shutdown_event.wait()
        # Automatically cleanup resources when shutdown occurs
        self._cleanup_resources()
    
    def shutdown(self) -> None:
        """
        Signal the ground station to shut down gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Shutdown requested")
        self._shutdown_event.set()

    def start(self) -> None:
        """
        Start the ground station.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Starting ground station")
        # Set socket timeout to make it non-blocking for shutdown
        self._sock.settimeout(self.socket_timeout)

def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for the ground station.
    
    Args:
        None
        
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Simple Payload Ground Station"
    )
    
    # Image saving
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save received detection images to disk"
    )
    parser.add_argument(
        "--image-save-dir",
        default="./images/",
        help="Directory to save received detection images",
        type=str
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        default="./logs/simple_payload_ground_station/",
        help="Directory to save log files with timestamps",
        type=str
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
        type=str
    )

    # Socket parameters
    parser.add_argument(
        "--socket-timeout",
        default=1.0,
        help="Socket timeout in seconds for UDP operations (non-negative float)",
        type=float
    )

    # UDP parameters
    parser.add_argument(
        "--udp-buffer-size",
        default=65536,
        help="UDP receive buffer size in bytes",
        type=int
    )
    parser.add_argument(
        "--udp-ip",
        default="0.0.0.0",
        help="UDP IP address to bind to (0.0.0.0 for all interfaces)",
        type=str
    )
    parser.add_argument(
        "--udp-port",
        default=5602,
        help="UDP port to listen on",
        type=int
    )
    
    return parser.parse_args()

def main() -> None:
    """
    Main entry point for the ground station application.
    
    Args:
        None
        
    Returns:
        None
    """
    args = get_args()
    ground_station = SimplePayloadGroundStation(args)
    
    try:
        ground_station.start()
        ground_station.run()
    except KeyboardInterrupt:
        ground_station.logger.info(
            "Received keyboard interrupt, shutting down..."
        )
        ground_station.shutdown()
    except Exception as e:
        ground_station.logger.error(f"Application error: {e}")
        ground_station.shutdown()

if __name__ == "__main__":
    main()
