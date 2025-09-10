import asyncio
import argparse
import base64
import json
import logging
import os
import piexif
import socket
import socketio
import threading
import time
import uvicorn

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
        
        # Image parameters
        self.image_save_dir = args.image_save_dir
        self.save_images = args.save_images
        # Socket parameters
        self.socket_timeout = args.socket_timeout
        # Socket.IO parameters
        self.socketio_enabled = args.socketio_enabled
        self.socketio_host = args.socketio_host
        self.socketio_port = args.socketio_port
        # UDP parameters
        self.udp_buffer_size = args.udp_buffer_size
        self.udp_ip = args.udp_ip
        self.udp_port = args.udp_port

        # Threads

        # Socket IO thread
        self._socketio_server_thread = None
        self._socketio_server_active = False
        # UDP receiver thread
        self._udp_receiver_thread = None
        self._udp_receiver_active = False

        # Events

        # Shutdown event for graceful termination
        self._shutdown_event = threading.Event()
        # Start event for initialization synchronization
        self._start_event = threading.Event()
        
        # Setup Socket.IO server
        self._sio_loop = None  # Store reference to async event loop
        self._sio_server = None
        self._sio_server_task = None
        if self.socketio_enabled:
            self._sio_server = socketio.AsyncServer(
                cors_allowed_origins="*",
                async_mode="asgi"
            )
            self._setup_socketio_events()
        # UDP socket (initialized in start())
        self._sock = None
        
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
    
    def _map_mode_to_status(self, mode: str) -> int:
        """
        Map drone mode to MAV_STATE status value.
        
        Args:
            mode: Drone mode string
            
        Returns:
            int: MAV_STATE status value
        """
        status_map = {
            "GUIDED": 4,      # MAV_STATE_ACTIVE
            "AUTO": 4,        # MAV_STATE_ACTIVE  
            "RTL": 4,         # MAV_STATE_ACTIVE
            "LAND": 4,        # MAV_STATE_ACTIVE
            "TAKEOFF": 4,     # MAV_STATE_ACTIVE
            "LOITER": 4,      # MAV_STATE_ACTIVE
            "MANUAL": 3,      # MAV_STATE_STANDBY
            "STABILIZE": 3,   # MAV_STATE_STANDBY
            "ALT_HOLD": 3,    # MAV_STATE_STANDBY
            "POSHOLD": 3,     # MAV_STATE_STANDBY
        }

        return status_map.get(mode, 1)  # Default to MAV_STATE_UNINIT if unknown

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
        # Broadcast telemetry to Socket.IO clients if enabled
        if self.socketio_enabled:
            self._broadcast_telemetry_to_clients(message)
    
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

    ##################################
    ### Socket.IO server functions ###
    ##################################
    
    async def _start_socketio_server(self) -> None:
        """
        Start the Socket.IO server with proper lifecycle management.
        
        Creates and runs a uvicorn ASGI server hosting the Socket.IO server.
        Handles graceful shutdown when _socketio_server_active is set to False.
        Implements proper asyncio server management patterns.
        """
        if not self.socketio_enabled or self._sio_server is None:
            return
            
        # Create ASGI app with Socket.IO server
        app = socketio.ASGIApp(self._sio_server, socketio_path="/ws/socket.io")
        # Configure uvicorn server with appropriate settings
        config = uvicorn.Config(
            app, 
            host=self.socketio_host,
            port=self.socketio_port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        self._logger.info(
            f"Starting Socket.IO server on {self.socketio_host}:"
            f"{self.socketio_port}"
        )
        
        try:
            # Start the server as a task
            server_task = asyncio.create_task(server.serve())
            
            # Monitor for shutdown signal
            while self._socketio_server_active:
                # Check if server task completed unexpectedly
                if server_task.done():
                    # Server stopped, check for exceptions
                    try:
                        await server_task
                    except Exception as e:
                        self._logger.error(f"Socket.IO server crashed: {e}")
                    break
                # Yield control to event loop
                await asyncio.sleep(0.1)
            
            # Initiate graceful shutdown
            if not server_task.done():
                self._logger.info("Initiating Socket.IO server shutdown...")
                server.should_exit = True
                
                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(server_task, timeout=3.0)
                    self._logger.info("Socket.IO server shutdown gracefully")
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "Socket.IO server shutdown timeout, "
                        "forcing cancellation"
                    )
                    server_task.cancel()
                    
                    # Wait for cancellation to complete
                    try:
                        await server_task
                    except asyncio.CancelledError:
                        self._logger.info("Socket.IO server task cancelled")
                    except Exception as e:
                        self._logger.error(
                            f"Error during server cancellation: {e}"
                        )
        except asyncio.CancelledError:
            # Handle task cancellation (e.g., from thread shutdown)
            self._logger.info("Socket.IO server task was cancelled")
            server.should_exit = True
            if not server_task.done():
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            self._logger.error(f"Unexpected error in Socket.IO server: {e}")
            # Ensure server is stopped on error
            if not server_task.done():
                server.should_exit = True
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
        finally:
            self._logger.info("Socket.IO server stopped")
    
    def _broadcast_telemetry_to_clients(self, message: dict[str, Any]) -> None:
        """
        Broadcast telemetry data to all connected Socket.IO clients.
        
        Converts drone telemetry format to webserver-compatible format and
        broadcasts to all connected Socket.IO clients using cross-thread
        communication via asyncio.run_coroutine_threadsafe().
        
        Args:
            message: Telemetry message from drone containing id, mode, armed,
                    location (lat/lon/alt), and attitude (yaw) fields
                    
        Returns:
            None
            
        Raises:
            KeyError: If required telemetry fields are missing from message
            Exception: Logs any broadcasting or parsing errors
        """
        if not self.socketio_enabled or self._sio_server is None:
            return
        
        try:
            drone_id = message['id']
            mode = message['mode']
            armed = message['armed']
            location = message['location']
            attitude = message['attitude']
            
            # Convert to webserver telemetry format
            webserver_telemetry = [{
                "drone_id": int(drone_id.split('_')[1]),
                "drone_name": str(drone_id),
                "status": self._map_mode_to_status(mode),
                "connected": True,
                "armed": armed,
                "guided": mode == "GUIDED",
                "manual_input": mode == "MANUAL",
                "position": {
                    "lat": location['lat'],
                    "lon": location['lon'], 
                    "alt": location['alt'],
                },
                "heading": attitude['yaw'],
            }]
            
            # Broadcast to all connected clients using cross-thread comm
            # Bridges the sync UDP thread with async Socket.IO server thread
            if self._sio_loop is not None and not self._sio_loop.is_closed():
                # Schedule the async emit operation on the Socket.IO event loop
                asyncio.run_coroutine_threadsafe(
                    self._sio_server.emit("telemetry", webserver_telemetry),
                    self._sio_loop
                )
                self._logger.debug(
                    f"Broadcast telemetry for drone {drone_id} to "
                    "Socket.IO clients"
                )
            else:
                self._logger.warning(
                    "Socket.IO event loop not available - "
                    "cannot broadcast telemetry"
                )
        except Exception as e:
            self._logger.error(
                f"Error broadcasting telemetry to Socket.IO clients: {e}"
            )

    def _setup_socketio_events(self) -> None:
        """
        Set up Socket.IO server event handlers.
        
        Configures async event handlers for client connect and disconnect events.
        These handlers log client connection/disconnection events for monitoring.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._sio_server is None:
            return
            
        @self._sio_server.event
        async def connect(sid, environ):
            self._logger.info(f"Socket.IO client connected: {sid}")
        @self._sio_server.event  
        async def disconnect(sid):
            self._logger.info(f"Socket.IO client disconnected: {sid}")
    
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
        self._stop_threads()
        if hasattr(self, '_sock') and self._sock:
            self._sock.close()
            self._logger.info("Closed UDP socket")
        self._shutdown_event.clear()
        self._logger.info("Ground station cleanup complete")

    def _start_socketio_server_thread(self) -> None:
        """
        Start the Socket.IO server thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If Socket.IO server thread is already started
        """
        if self._socketio_server_thread is not None:
            raise RuntimeError("Socket.IO server thread already started")
        
        self._socketio_server_active = True
        self._socketio_server_thread = threading.Thread(
            target=self._socketio_server_worker,
            daemon=True,
            name="SocketIO-Server"
        )
        self._socketio_server_thread.start()
        self._logger.info("Started Socket.IO server thread")

    def _stop_socketio_server_thread(self) -> None:
        """
        Stop the Socket.IO server thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._socketio_server_thread is not None:
            self._logger.info("Stopping Socket.IO server...")
            self._socketio_server_active = False
            self._socketio_server_thread.join(timeout=5.0)
            if self._socketio_server_thread.is_alive():
                self._logger.warning(
                    "Socket.IO server thread did not stop gracefully"
                )
            else:
                self._logger.info("Socket.IO server thread stopped")
            self._socketio_server_thread = None

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
    
    def _start_threads(self) -> None:
        """
        Start all worker threads.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Starting threads...")
        
        self._start_udp_receiver()
        if self.socketio_enabled:
            self._start_socketio_server_thread()
        
        self._logger.info("All threads started successfully")

    def _stop_threads(self) -> None:
        """
        Stop all worker threads gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Stopping threads...")
        
        self._stop_udp_receiver()
        if self.socketio_enabled:
            self._stop_socketio_server_thread()
        
        self._logger.info("All threads stopped successfully")

    ###############################
    ### Thread worker functions ###
    ###############################
    
    def _socketio_server_worker(self) -> None:
        """
        Worker thread function that runs the Socket.IO server.
        
        Creates a new asyncio event loop for the thread and runs the Socket.IO
        server within it. Stores the event loop reference for cross-thread
        communication and ensures proper cleanup on shutdown.
        
        Args:
            None
            
        Returns:
            None
            
        Note:
            This function blocks until the server stops or encounters an error.
            The event loop is isolated to this thread to avoid conflicts with
            the main application thread.
        """
        self._logger.info("Socket.IO server worker started")
        
        # Wait for initialization to complete
        self._start_event.wait()
        
        def run_server():
            """
            Run the asyncio event loop for the Socket.IO server.
            
            Creates and manages the event loop lifecycle, storing a reference
            for cross-thread communication and ensuring proper cleanup.
            """
            # Create isolated event loop for this thread
            self._sio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._sio_loop)
            try:
                # Run the async server until completion or cancellation
                self._sio_loop.run_until_complete(self._start_socketio_server())
            except Exception as e:
                self._logger.error(f"Socket.IO server error: {e}")
            finally:
                # Clean up event loop and clear cross-thread reference
                self._sio_loop.close()
                self._sio_loop = None  # Prevents further cross-thread calls
        
        run_server()
        self._logger.info("Socket.IO server worker stopped")

    def _udp_receiver_worker(self) -> None:
        """
        Worker thread function that receives and processes UDP messages.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("UDP receiver worker started")
        
        # Wait for initialization to complete
        self._start_event.wait()
        
        # Check if socket is properly initialized
        if self._sock is None:
            self._logger.error(
                "UDP socket not initialized, cannot start receiver"
            )
            return
            
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
        
        # Setup UDP socket
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((self.udp_ip, self.udp_port))
            # Set socket timeout to make it non-blocking for shutdown
            self._sock.settimeout(self.socket_timeout)
            self._logger.info(
                f"UDP socket bound to {self.udp_ip}:{self.udp_port}"
            )
        except OSError as e:
            self._logger.error(
                f"Failed to bind UDP socket to {self.udp_ip}:"
                f"{self.udp_port}: {e}"
            )
            if self._sock:
                self._sock.close()
                self._sock = None
            raise RuntimeError(f"UDP socket initialization failed: {e}") from e
        
        self._start_threads()
        self._start_event.set()

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
    
    # Directories
    parser.add_argument(
        "--image-save-dir",
        default="./images/",
        help="Directory to save received detection images",
        type=str
    )

    # Image parameters
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save received detection images to disk"
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

    # Socket.IO telemetry server
    parser.add_argument(
        "--socketio-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Socket.IO server for telemetry broadcasting"
    )
    parser.add_argument(
        "--socketio-host",
        default="0.0.0.0",
        help="Socket.IO server host address",
        type=str
    )
    parser.add_argument(
        "--socketio-port",
        default=8005,
        help="Socket.IO server port",
        type=int
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
