import argparse
import base64
import cv2
import json
import math
import numpy as np
import libcamera
import logging
import os
import queue
import time
import threading
import socket
import json

from dataclasses import dataclass
from datetime import datetime
from dronekit import connect, VehicleMode, LocationGlobalRelative
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection
)
from picamera2.devices.imx500.postprocess import scale_boxes
from rpi_hardware_pwm import HardwarePWM
from typing import Any

@dataclass
class Detection:
    box: tuple[int, int, int, int]
    category: int
    conf: float
        
class SimplePayloadDrone:
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
        log_filename = f"target_detection_{timestamp}.log"
        log_filepath = os.path.join(args.log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging._nameToLevel.get(args.log_level, "DEBUG"))
        self._logger.addHandler(file_handler)
        
        # Get model path
        self.model_type = args.model_type
        model_rpk_name = ""
        if args.model_type == "yolo8n":
            model_rpk_name = "imx500_network_yolov8n_pp.rpk"
        elif args.model_type == "yolo11n":
            model_rpk_name = "imx500_network_yolo11n_pp.rpk"
        elif args.model_type == "efficientdet_lite0":
            model_rpk_name = "imx500_network_efficientdet_lite0_pp.rpk"
        elif args.model_type == "nanodet_plus":
            model_rpk_name = "imx500_network_nanodet_plus_416x416.rpk"
        elif args.model_type == "nanodet_plus_pp":
            model_rpk_name = \
                "imx500_network_nanodet_plus_416x416_pp.rpk"
        elif args.model_type == "ssd_mobilenetv2_fpnlite":
            model_rpk_name = \
                "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
        model_path = os.path.join(args.model_dir, model_rpk_name)
        # Get label path
        self.labels = args.labels
        labels_name = ""
        if args.labels == "coco":
            labels_name = "coco_labels.txt"
        labels_path = os.path.join(args.labels_dir, labels_name)
        
        # Initialize IMX500
        self._imx500 = IMX500(model_path)
        # Create intrinsics
        self._intrinsics = self._imx500.network_intrinsics
        if not self._intrinsics:
            self._logger.info("No intrinsics provided")
            self._intrinsics = NetworkIntrinsics()
            self._intrinsics.task = "object detection"
        if self._intrinsics.task != "object detection":
            raise ValueError(
                f"Model type {self.model_type} cannot be used for OD"
            )
        # Update intrinsics
        if args.use_default_intrinsics:
            self._intrinsics.update_with_defaults()
        else:
            self._intrinsics.bbox_normalization = args.bbox_normalization
            self._intrinsics.bbox_order = args.bbox_order
            self._intrinsics.fps = args.fps
            self._intrinsics.ignore_dash_labels = args.ignore_dash_labels
            self._intrinsics.inference_rate = args.inference_rate
            with open(labels_path, "r") as f:
                self._intrinsics.labels = f.read().splitlines()
            self._intrinsics.postprocess = args.postprocess
            self._intrinsics.preserve_aspect_ratio = \
                args.preserve_aspect_ratio
            self._intrinsics.softmax = args.softmax
        # Labels used
        if self._intrinsics.ignore_dash_labels:
            # Dash removed labels
            self._dash_rmv_labels = [
                label for label in self._intrinsics.labels \
                if label and label != "-"
            ]
            self._labels_used = self._dash_rmv_labels
        else:
            self._labels_used = self._intrinsics.labels
        
        # Initialize camera
        self._picam2 = Picamera2(self._imx500.camera_num)
        
        # Parameters

        # Camera parameters
        self.buffer_count = args.buffer_count
        # Connection parameters
        self.rpi_baud_rate = args.rpi_baud_rate
        self.rpi_serial_port = args.rpi_serial_port
        self.udp_image_quality = args.udp_image_quality
        self.udp_ip = args.udp_ip
        self.udp_port = args.udp_port
        self.udp_pub = args.udp_pub
        self.udp_queue_block = args.udp_queue_block
        self.udp_queue_timeout = args.udp_queue_timeout
        self.udp_queue_maxsize = args.udp_queue_maxsize
        # Debug parameters
        self.debug_camera = args.debug_camera
        self.debug_detect = args.debug_detect
        self.debug_detect_no_vech = args.debug_detect_no_vech
        self.debug_goto_waypoints = args.debug_goto_waypoints
        # Detection parameters
        self.detect_classes = args.detect_classes
        if len(self.detect_classes) == 0:
            # No interested classes -> All classes
            self.detect_classes = [i for i in range(len(self._labels_used))]
        self.hflip = args.hflip
        self.iou = args.iou
        self.max_detections = args.max_detections
        self.threshold = args.threshold
        self.vflip = args.vflip
        # Drone parameters
        self.drone_id = args.drone_id
        # Waypoint parameters
        self.wpt_json_dir = args.wpt_json_dir
        self.wpt_json_filename = args.wpt_json_filename
        self.wpt_json_filepath = os.path.join(
            self.wpt_json_dir, 
            self.wpt_json_filename
        )
        self.wpt_arrival_threshold = args.wpt_arrival_threshold
        
        # Threads
        
        # Camera pitch worker thread
        self._camera_pitch_thread = None
        self._camera_pitch_active = False
        # Detection worker thread
        self._detection_thread = None
        self._detection_active = False
        # Goto waypoints thread
        self._goto_waypoints_thread = None
        self._goto_waypoints_active = False
        # UDP publishing worker thread 
        self._message_queue = queue.Queue(maxsize=args.udp_queue_maxsize)
        self._udp_thread = None
        self._sentinel = object()  # Unique sentinel object for shutdown
        # Shutdown event for graceful termination
        self._shutdown_event = threading.Event()
        # Start event for synchronized thread startup
        self._start_event = threading.Event()
        
        # Setup comms
        if not args.debug_detect_no_vech:
            self._vehicle=connect(
                self.rpi_serial_port, 
                wait_ready=True, 
                baud=self.rpi_baud_rate
            )
        else:
            self._vehicle = None
        # Setup servo
        self._pwm = HardwarePWM(0, 50) # 50 Hz for servo
        # Setup UDP socket
        if args.udp_pub:
            self._sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Setup waypoints from json
        self._wpt_goto_list = self._parse_json_to_waypoints(
            self.wpt_json_filepath
        )
        if not self._wpt_goto_list:
            raise RuntimeError(
                f"Error loading waypoints from {self.wpt_json_filepath}"
            )

        # Store detections with thread safety
        self._detections_lock = threading.Lock()
        self._last_detections: list[Detection] = []
    
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
    
    ##################################
    ### Detection Helper Functions ###
    ##################################
    
    def _encode_detection_message(
            self, 
            detections: list[Detection], 
            image: np.ndarray,
            location: dict[str, Any],
        ) -> bytes:
        """
        Encode detections and image into bytes for UDP transmission.
        
        Args:
            detections: List of Detection objects to encode
            image: NumPy array representing the image
            location: Dictionary containing location data (lat, lon, alt)
            
        Returns:
            bytes: JSON-encoded message as bytes for UDP transmission
        """
        # Convert BGR to RGB for correct color encoding
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Encode image to JPEG bytes with configurable quality
        _, img_encoded = cv2.imencode(
            '.jpg', 
            image_rgb, 
            [cv2.IMWRITE_JPEG_QUALITY, self.udp_image_quality]
        )
        jpeg_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        self._logger.debug(
            f"Image encoding: JPEG={len(jpeg_bytes):,} bytes, "
            f"Base64={len(img_base64):,} bytes, "
            f"Quality={self.udp_image_quality}"
        )
        
        # Convert detections to serializable format
        detection_data = []
        
        for d in detections:
            detection_data.append({
                'box': d.box, # tuple -> JSON array automatically
                'category': d.category, # int
                'confidence': float(d.conf), # Convert numpy float32 to float
                'label': self._labels_used[d.category], # str
            })
        
        # Create message payload
        message = {
            'detections': detection_data,
            'image': img_base64,
            'location': location,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Convert to JSON and encode as bytes
        message_bytes = json.dumps(message).encode('utf-8')
        self._logger.debug(
            f"Total UDP message size: {len(message_bytes):,} bytes"
        )
        
        return message_bytes
    
    def _parse_detections(
            self, 
            request
        ) -> tuple[list[Detection], np.ndarray | None]:
        """
        Parse the output tensor into a number of detected objects, scaled to the
        ISP output.
        
        Args:
            request: Picamera2 request object with both metadata and image data
            
        Returns:
            tuple: (list of Detection objects, image array or None)
        """
        metadata = request.get_metadata()
        np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return [], None
        input_w, input_h = self._imx500.get_input_size()
        
        if self._intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(
                    outputs=np_outputs[0], 
                    conf=self.threshold, 
                    iou_thres=self.iou,
                    max_out_dets=self.max_detections
                )[0]
            boxes = scale_boxes(
                boxes, 
                1, 
                1, 
                input_h, 
                input_w, 
                False, 
                False
            )
        else:
            # boxes (num_class, 4)
            # scores (num_class, )
            # classes (num_class, )
            boxes = np_outputs[0][0]
            scores = np_outputs[1][0]
            classes = np_outputs[2][0]
            if self._intrinsics.bbox_normalization:
                # Assumes input_h == input_w
                boxes = boxes / input_h

            if self._intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)
        
        new_detections = [
            Detection(
                self._imx500.convert_inference_coords(
                    box, 
                    metadata,
                    self._picam2
                ),
                int(category),
                score,
            )
            for box, score, category in zip(boxes, scores, classes)
            if score > self.threshold
        ]
        
        return new_detections, request.make_array("main")

    ################################
    ### Drawing helper functions ###
    ################################

    def _draw_detections(self, request, stream: str = "main") -> None:
        """
        Draw the detections for this request onto the ISP output.
        
        Args:
            request: Picamera2 request object
            stream: Stream name to draw on (default "main")
            
        Returns:
            None
        """
        with self._detections_lock:
            detections_to_draw = self._last_detections.copy()
        if not detections_to_draw:
            return
            
        with MappedArray(request, stream) as m:
            # Draw all detections using the shared helper function
            for detection in detections_to_draw:
                self._draw_single_detection(
                    detection, 
                    m.array, 
                    self._labels_used
                )
            # Draw ROI if preserve aspect ratio is enabled
            if self._intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self._imx500.get_roi_scaled(request)
                color = (255, 0, 0) # red
                cv2.putText(
                    m.array, 
                    "ROI", 
                    (b_x + 5, b_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color,
                    1
                )
                cv2.rectangle(
                    m.array, 
                    (b_x, b_y), 
                    (b_x + b_w, b_y + b_h), 
                    (255, 0, 0, 0)
                )

    def _draw_detections_on_image(
            self,
            detections: list[Detection],
            image: np.ndarray,
        ) -> np.ndarray:
        """
        Helper function to draw detection bounding boxes and labels on an image.
        
        Args:
            detections: List of Detection objects to draw
            image: NumPy array representing the image
            
        Returns:
            np.ndarray: Copy of the image with detections drawn
        """
        if not detections:
            return image
        
        # Create a copy to draw on
        result_image = image.copy()
        
        # Draw all detections using the helper
        for detection in detections:
            self._draw_single_detection(
                detection, 
                result_image, 
                self._labels_used
            )

        return result_image

    def _draw_single_detection(
            self, 
            detection: Detection, 
            image: np.ndarray, 
            labels: list[str]
        ) -> None:
        """
        Helper function to draw a single detection box and label on an image.
        
        Args:
            detection: Detection object to draw
            image: NumPy array representing the image (modified in-place)
            labels: List of label strings
            
        Returns:
            None
        """
        x, y, w, h = detection.box
        label = f"{labels[detection.category]} ({detection.conf:.2f})"

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            1
        )
        text_x = x + 5
        text_y = y + 15

        # Create overlay for semi-transparent background
        overlay = image.copy()

        # Draw the background rectangle on the overlay
        cv2.rectangle(
            overlay,
            (text_x, text_y - text_height),
            (text_x + text_width, text_y + baseline),
            (255, 255, 255), # White background color
            cv2.FILLED
        )

        alpha = 0.30
        cv2.addWeighted(
            overlay, 
            alpha, 
            image, 
            1 - alpha, 
            0, 
            image
        )

        # Draw text on top of the background
        cv2.putText(
            image, 
            label, 
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1
        )

        # Draw detection box
        cv2.rectangle(
            image, 
            (x, y), 
            (x + w, y + h), 
            (0, 255, 0, 0), 
            thickness=2
        )

    #######################################
    ### Goto waypoints helper functions ###
    #######################################

    @staticmethod
    def _get_distance_metres(aLocation1, aLocation2) -> float:
        """
        Return the ground distance in metres between two LocationGlobal objects.

        This method is an approximation, and will not be accurate over large 
        distances and close to the earth's poles. It comes from the ArduPilot 
        test code.
        
        Args:
            aLocation1: First LocationGlobal object
            aLocation2: Second LocationGlobal object
            
        Returns:
            float: Distance in metres between the two locations
        """
        dlat = aLocation2.lat - aLocation1.lat
        dlong = aLocation2.lon - aLocation1.lon
        return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5
    
    def _goto_waypoint(self, targetLocation: LocationGlobalRelative) -> None:
        """
        Navigate vehicle to a specific waypoint using DroneKit simple_goto.
        
        Commands the vehicle to navigate to the specified target location and
        monitors progress until the waypoint is reached or the vehicle exits
        GUIDED mode. Uses a distance-based approach to determine arrival.
        
        Args:
            targetLocation: Target waypoint as LocationGlobalRelative object
            
        Returns:
            None
            
        Raises:
            RuntimeError: If vehicle is None or not available
            
        Note:
            This function blocks until the waypoint is reached or the vehicle
            exits GUIDED mode. The arrival threshold is configurable via
            wpt_arrival_threshold parameter (default 1% of initial distance)
            to handle GPS accuracy limitations.
        """
        # Validate vehicle availability
        if self._vehicle is None:
            self._goto_waypoints_active = False
            raise RuntimeError("Vehicle is None")
        # Get current position and calculate initial distance
        currentLocation = self._vehicle.location.global_relative_frame
        targetDistance = self._get_distance_metres(
            currentLocation, 
            targetLocation
        )
        # Command vehicle to navigate to target
        self._vehicle.simple_goto(targetLocation)
        self._logger.info(f"Going to target location {targetLocation}")

        # Monitor progress until arrival or shutdown
        while self._goto_waypoints_active and not self._shutdown_event.is_set():
            # Check if vehicle is still in GUIDED mode
            current_mode = self._vehicle.mode.name
            if current_mode != "GUIDED":
                self._logger.warning(
                    f"Waypoint navigation stopped - "
                    f"vehicle changed to {current_mode} mode"
                )
                self._goto_waypoints_active = False
                break
            # Calculate remaining distance to target
            remainingDistance = self._get_distance_metres(
                self._vehicle.location.global_relative_frame, 
                targetLocation
            )
            self._logger.debug(f"Distance to target: {remainingDistance:.2f}m")
            # Check if we've reached the target using arrival threshold
            if remainingDistance <= targetDistance * self.wpt_arrival_threshold:
                self._logger.debug(f"Reached target {targetLocation}")
                break
            time.sleep(0.5) # Wait before next distance check
        
        # Log reason for exit if interrupted by shutdown
        if not self._goto_waypoints_active: 
            self._logger.warning(
                f"Waypoint navigation to {targetLocation} interuppted"
            )
        elif self._shutdown_event.is_set():
            self._logger.info("Waypoint navigation interrupted by shutdown")

    def _parse_json_to_waypoints(
            self, 
            filepath: str
        ) -> list[LocationGlobalRelative] | None:
        """
        Parse coordinate data from a JSON file into LocationGlobalRelative 
        objects for the current drone.
        
        Reads a JSON file containing waypoint data organized by drone_id with 
        lat/lon/alt coordinates and converts them into DroneKit 
        LocationGlobalRelative objects for navigation use.
        
        Args:
            filepath: Path to the JSON file containing waypoint data
        
        Returns:
            list[LocationGlobalRelative] | None: List of waypoint objects if 
                successful, None if file not found, invalid JSON, drone_id not 
                found, or other errors occur
        
        Raises:
            None: All exceptions are caught and logged, returning None on 
                failure
        
        Expected JSON format:
            {
                "drone_0": [
                    {"lat": float, "lon": float, "alt": float},
                    {"lat": float, "lon": float, "alt": float},
                    ...
                ],
                "drone_1": [...]
            }
        """
        try:
            waypoint_objects: list[LocationGlobalRelative] = []
            with open(filepath, 'r') as file:
                data = json.load(file)
            # Validate data structure - must be a dict with drone_id keys
            if not isinstance(data, dict):
                self._logger.error(
                    f"Invalid JSON structure in {filepath}: expected dict, "
                    f"got {type(data).__name__}"
                )
                return None
            
            # Check if current drone_id exists in the data
            if self.drone_id not in data:
                self._logger.error(
                    f"Drone ID '{self.drone_id}' not found in {filepath}. "
                    f"Available drone IDs: {list(data.keys())}"
                )
                return None
            
            # Get waypoint list for current drone
            drone_waypoints = data[self.drone_id]
            if not isinstance(drone_waypoints, list):
                self._logger.error(
                    f"Invalid waypoint data for drone '{self.drone_id}' in "
                    f"{filepath}: expected list, got "
                    f"{type(drone_waypoints).__name__}"
                )
                return None
            
            # Parse each waypoint dictionary into LocationGlobalRelative object
            for i, waypoint_data in enumerate(drone_waypoints):
                if not isinstance(waypoint_data, dict):
                    self._logger.error(
                        f"Invalid waypoint at index {i} for drone "
                        f"'{self.drone_id}' in {filepath}: expected dict, got "
                        f"{type(waypoint_data).__name__}"
                    )
                    return None
                # Validate required keys
                required_keys = {'lat', 'lon', 'alt'}
                if not required_keys.issubset(waypoint_data.keys()):
                    missing_keys = required_keys - waypoint_data.keys()
                    self._logger.error(
                        f"Missing required keys {missing_keys} in waypoint "
                        f"{i} for drone '{self.drone_id}' from {filepath}"
                    )
                    return None
                try:
                    lat = float(waypoint_data['lat'])
                    lon = float(waypoint_data['lon'])
                    alt = float(waypoint_data['alt'])
                    # Validate latitude range (-90 to 90)
                    if not (-90.0 <= lat <= 90.0):
                        raise ValueError(
                            f"Invalid latitude {lat}: must be between "
                            f"-90 and 90 degrees"
                        )
                    # Validate longitude range (-180 to 180)
                    if not (-180.0 <= lon <= 180.0):
                        raise ValueError(
                            f"Invalid longitude {lon}: must be between "
                            f"-180 and 180 degrees"
                        )
                    waypoint = LocationGlobalRelative(lat, lon, alt)
                    waypoint_objects.append(waypoint)
                except (ValueError, TypeError) as e:
                    self._logger.error(
                        f"Invalid coordinate values in waypoint {i} for drone "
                        f"'{self.drone_id}' from {filepath}: {e}"
                    )
                    return None
            
            if not waypoint_objects:
                self._logger.warning(
                    f"No valid waypoints parsed for drone '{self.drone_id}' "
                    f"from {filepath}"
                )
                return None
            self._logger.info(
                f"Successfully parsed {len(waypoint_objects)} waypoints for "
                f"drone '{self.drone_id}' from {filepath}"
            )

            return waypoint_objects
        except FileNotFoundError:
            self._logger.error(f"Waypoints file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON format in {filepath}: {e}")
            return None
        except Exception as e:
            self._logger.error(
                f"Unexpected error parsing waypoints from {filepath}: {e}"
            )
            return None

    ######################################
    ### Message queue helper functions ###
    ######################################

    def _dequeue_message(self) -> bytes | None:
        """
        Helper function to dequeue a message with proper error handling.
        
        Args:
            None
            
        Returns:
            bytes | None: Message from queue or None if error/timeout
        """
        try:
            message = self._message_queue.get(
                block=self.udp_queue_block, 
                timeout=self.udp_queue_timeout
            )
            self._logger.debug("Dequeued message from UDP queue")
            return message
        except queue.Empty:
            # self._logger.debug(
            #     "No message available in UDP queue (timeout/empty)"
            # )
            return None
        except Exception as e:
            self._logger.debug(f"Failed to dequeue message: {e}")
            return None

    def _enqueue_message(self, message: bytes) -> None:
        """
        Helper function to enqueue a message with proper error handling.
        
        Args:
            message: Bytes message to enqueue for UDP transmission
            
        Returns:
            None
        """
        try:
            self._message_queue.put(
                message, 
                block=self.udp_queue_block, 
                timeout=self.udp_queue_timeout
            )
            self._logger.debug("Queued detection message for UDP publishing")
        except queue.Full:
            # self._logger.debug(
            #     "Message queue is full, dropping detection message"
            # )
            return None
        except Exception as e:
            self._logger.debug(f"Failed to queue detection message: {e}")
            return None
    
    def _enqueue_sentinel(self) -> None:
        """
        Enqueue the sentinel object to signal shutdown to UDP worker thread.
        
        Args:
            None
            
        Returns:
            None
        """
        # Always block indefinitely for sentinel to ensure shutdown happens
        self._logger.debug(
            "Queueing sentinel object to signal UDP worker shutdown"
        )
        self._message_queue.put(self._sentinel, block=True, timeout=None)
        self._logger.debug("Queued sentinel object for UDP worker shutdown")

    ################################
    ### Vehicle helper functions ###
    ################################
    def _get_vehicle_location(self) -> dict[str, Any]:
        """
        Get current vehicle location if vehicle is connected.
        
        Args:
            None
            
        Returns:
            dict[str, Any]: Location with lat/lon/alt keys (values may be None)
        """
        if self._vehicle is None:
            return {
                'lat': None,
                'lon': None,
                'alt': None
            }
            
        location = self._vehicle.location.global_frame
        return {
            'lat': location.lat,
            'lon': location.lon,
            'alt': location.alt
        }

    ################################
    ### Thread handler functions ###
    ################################

    def _start_camera_pitch_worker(self) -> None:
        """
        Start the camera pitch worker thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If camera pitch worker is already started
        """
        if self._camera_pitch_thread is not None:
            raise RuntimeError("Camera pitch worker already started")
        
        self._camera_pitch_active = True
        self._camera_pitch_thread = threading.Thread(
            target=self._camera_pitch_worker,
            daemon=True,
            name="Camera-Pitch-Worker"
        )
        self._camera_pitch_thread.start()
        self._logger.info("Started camera pitch worker thread")

    def _stop_camera_pitch_worker(self) -> None:
        """
        Stop the camera pitch worker thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._camera_pitch_thread is not None:
            self._logger.info("Stopping camera pitch worker...")
            self._camera_pitch_active = False
            self._camera_pitch_thread.join(timeout=5.0)
            if self._camera_pitch_thread.is_alive():
                self._logger.warning(
                    "Camera pitch worker thread did not stop gracefully"
                )
            else:
                self._logger.info("Camera pitch worker thread stopped")
            self._camera_pitch_thread = None

    def _start_detection_worker(self) -> None:
        """
        Start the detection worker thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If detection worker is already started
        """
        if self._detection_thread is not None:
            raise RuntimeError("Detection worker already started")
        
        self._detection_active = True
        self._detection_thread = threading.Thread(
            target=self._detection_worker,
            daemon=True,
            name="Detection-Worker"
        )
        self._detection_thread.start()
        self._logger.info("Started detection worker thread")

    def _stop_detection_worker(self) -> None:
        """
        Stop the detection worker thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._detection_thread is not None:
            self._logger.info("Stopping detection worker...")
            self._detection_active = False
            self._detection_thread.join(timeout=5.0)
            if self._detection_thread.is_alive():
                self._logger.warning(
                    "Detection worker thread did not stop gracefully"
                )
            else:
                self._logger.info("Detection worker thread stopped")
            self._detection_thread = None

    def _start_goto_waypoints_worker(self) -> None:
        """
        Start the goto waypoints worker thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If goto waypoints worker is already started
        """
        if self._goto_waypoints_thread is not None:
            raise RuntimeError("Goto waypoints worker already started")
        
        self._goto_waypoints_active = True
        self._goto_waypoints_thread = threading.Thread(
            target=self._goto_waypoints_worker,
            daemon=True,
            name="Goto-Waypoints-Worker"
        )
        self._goto_waypoints_thread.start()
        self._logger.info("Started goto waypoints worker thread")

    def _stop_goto_waypoints_worker(self) -> None:
        """
        Stop the goto waypoints worker thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._goto_waypoints_thread is not None:
            self._logger.info("Stopping goto waypoints worker...")
            self._goto_waypoints_active = False
            self._goto_waypoints_thread.join(timeout=5.0)
            if self._goto_waypoints_thread.is_alive():
                self._logger.warning(
                    "Goto waypoints worker thread did not stop gracefully"
                )
            else:
                self._logger.info("Goto waypoints worker thread stopped")
            self._goto_waypoints_thread = None

    def _start_udp_publisher(self) -> None:
        """
        Start the UDP publisher thread.
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            RuntimeError: If UDP publisher is already started
        """
        if self._udp_thread is not None:
            raise RuntimeError("UDP publisher already started")
        
        self._udp_thread = threading.Thread(
            target=self._udp_publisher_worker,
            daemon=True,
            name="UDP-Publisher"
        )
        self._udp_thread.start()
        self._logger.info("Started UDP publisher thread")
    
    def _stop_udp_publisher(self) -> None:
        """
        Stop the UDP publisher thread gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        if self._udp_thread is not None:
            self._logger.info("Stopping UDP publisher...")
            self._enqueue_sentinel()
            self._udp_thread.join(timeout=5.0)
            if self._udp_thread.is_alive():
                self._logger.warning(
                    "UDP publisher thread did not stop gracefully"
                )
            else:
                self._logger.info("UDP publisher thread stopped")
            self._udp_thread = None
    
    def _start_threads(self) -> None:
        """
        Start all required threads based on operation mode.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Starting threads...")
        
        if self.debug_camera:
            self._start_camera_pitch_worker()
        if self.debug_detect or self.debug_detect_no_vech:
            self._start_detection_worker()
        if self.debug_goto_waypoints:
            self._start_goto_waypoints_worker()
        if not (
            self.debug_camera or 
            self.debug_detect or 
            self.debug_detect_no_vech or 
            self.debug_goto_waypoints
        ):
            self._start_camera_pitch_worker()
            self._start_detection_worker()
            self._start_goto_waypoints_worker()
        
        if self.udp_pub:
            self._start_udp_publisher()
        
        self._logger.info("All threads started successfully")

    def _stop_threads(self) -> None:
        """
        Stop all running threads gracefully.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Stopping threads...")
        
        # Stop camera pitch worker if running
        if self._camera_pitch_thread is not None:
            self._stop_camera_pitch_worker()
        
        # Stop detection worker if running
        if self._detection_thread is not None:
            self._stop_detection_worker()
        
        # Stop goto waypoints worker if running
        if self._goto_waypoints_thread is not None:
            self._stop_goto_waypoints_worker()

        # Stop UDP publisher if running
        if self._udp_thread is not None:
            self._stop_udp_publisher()
        
        self._logger.info("All threads stopped")

    ###############################
    ### Thread worker functions ###
    ###############################

    def _camera_pitch_worker(self) -> None:
        """
        Worker thread function that handles camera pitch correction with servo.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Camera pitch worker started")
        
        # Wait for synchronized start signal
        self._start_event.wait()
        self._logger.info("Camera pitch worker ready to begin processing")
        
        if self._vehicle is None:
            self._camera_pitch_active = False
            raise RuntimeError("Vehicle is None")
        
        while self._camera_pitch_active and not self._shutdown_event.is_set():
            try:
                pitch = math.degrees(self._vehicle.attitude.pitch)
                self._logger.debug(f"Pitch: {pitch:.2f} degrees")
                # Compensate
                servo_angle = self.base_angle + pitch
                # Limit servo movement
                servo_angle = max(min(servo_angle, 90), 0)
                # Convert angle to duty cycle
                duty_cycle = 5 + (servo_angle / 90) * 5  
                self._logger.debug(
                    f"Servo angle: {servo_angle:.1f}°, Duty: {duty_cycle:.2f}%"
                )
                self._pwm.change_duty_cycle(duty_cycle)
                time.sleep(0.02) # 20 ms update
            except Exception as e:
                self._logger.error(f"Camera pitch worker error: {e}")
        
        self._logger.info("Camera pitch worker stopped")

    def _detection_worker(self) -> None:
        """
        Worker thread function that continuously runs detection processing.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Detection worker started")
        
        # Wait for synchronized start signal
        self._start_event.wait()
        self._logger.info("Detection worker ready to begin processing")
        
        while self._detection_active and not self._shutdown_event.is_set():
            try:
                # Get the latest detections with synchronized image capture
                request = self._picam2.capture_request()
                current_detections, current_image = \
                    self._parse_detections(request)
                request.release()  # Always release the request

                # Store detections safely for preview drawing
                if current_detections and current_image is not None:
                    with self._detections_lock:
                        self._last_detections = current_detections
                else:
                    continue
                    
                # Get vehicle location for relevant detections
                vehicle_location = self._get_vehicle_location()
                
                # Filter relevant detections
                relevant_detections = []
                for d in current_detections:
                    if d.category in self.detect_classes:
                        relevant_detections.append(d)
                        # Create debug message with fixed format location
                        lat = vehicle_location['lat']
                        lng = vehicle_location['lon'] 
                        alt = vehicle_location['alt']
                        msg = (
                            f"{self._intrinsics.labels[d.category]} "
                            f"detected with {d.conf:.2f} confidence "
                            f"at lat:{lat}, lng:{lng}, alt:{alt}"
                        )
                        self._logger.debug(msg)
                
                # Send to message queue for UDP publishing if enabled
                if self.udp_pub and relevant_detections and \
                    current_image is not None:
                    # Draw detections on captured image
                    annotated_image = self._draw_detections_on_image(
                        relevant_detections,
                        current_image
                    )
                    # Encode message with detections, image, and location
                    message_bytes = self._encode_detection_message(
                        relevant_detections, 
                        annotated_image,
                        vehicle_location
                    )
                    # Add to message queue based on blocking configuration
                    self._enqueue_message(message_bytes)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1) # 10Hz detection rate
            except Exception as e:
                self._logger.error(f"Detection worker error: {e}")
                time.sleep(0.1) # Brief pause on error
        
        self._logger.info("Detection worker stopped")

    def _goto_waypoints_worker(self) -> None:
        """
        Worker thread function to make vehicle go to a series of waypoints.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Goto waypoints worker started")
        
        # Wait for synchronized start signal
        self._start_event.wait()
        self._logger.info("Goto waypoints worker ready to begin processing")

        if self._vehicle is None:
            self._goto_waypoints_active = False
            raise RuntimeError("Vehicle is None")
        if self._wpt_goto_list is None:
            self._goto_waypoints_active = False
            raise RuntimeError("Goto waypoints is None")

        # Wait for vehicle to be ready (in GUIDED mode)
        self._logger.info(
            "Waiting for vehicle to be ready for waypoint navigation..."
        )
        while self._goto_waypoints_active and not self._shutdown_event.is_set():
            current_mode = self._vehicle.mode.name
            if current_mode == "GUIDED":
                self._logger.info("Vehicle is ready - in GUIDED mode")
                break
            else:
                self._logger.debug(
                    f"Waiting for GUIDED mode (currently {current_mode})"
                )
                time.sleep(0.5) # Check every 500ms
        
        # Execute waypoint navigation
        self._logger.info(
            f"Starting navigation through {len(self._wpt_goto_list)} waypoints"
        )
        for i, wp in enumerate(self._wpt_goto_list):
            # Check for shutdown between waypoints
            if not self._goto_waypoints_active or self._shutdown_event.is_set():
                self._logger.info(f"Navigation interrupted at waypoint {i+1}")
                break
            try:
                self._logger.info(
                    f"Navigating to waypoint {i+1}/{len(self._wpt_goto_list)}: "
                    f"{wp}"
                )
                self._goto_waypoint(wp)
                self._logger.info(
                    f"Reached waypoint {i+1}/{len(self._wpt_goto_list)}"
                )
            except Exception as e:
                self._logger.error(f"Error navigating to waypoint {i+1}: {e}")
                # Stop waypoint mission on error for safety
                self._goto_waypoints_active = False
                break

        # Log termination reason based on current state
        if not self._goto_waypoints_active:
            # Active flag was set to False due to error
            self._logger.info(
                "Goto waypoints worker terminated - "
                "stopped due to navigation error"
            )
        elif self._shutdown_event.is_set():
            # Shutdown was requested
            self._logger.info(
                "Goto waypoints worker terminated - "
                "interrupted by shutdown event"
            )
        else:
            # Normal completion - finished all waypoints
            self._logger.info(
                "Goto waypoints worker completed - "
                "all waypoints reached successfully"
            )
            self._goto_waypoints_active = False
        self._logger.info("Goto waypoints worker stopped")

    def _udp_publisher_worker(self) -> None:
        """
        Worker thread function to process message queue and send UDP messages.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("UDP publisher worker started")
        
        # Wait for synchronized start signal
        self._start_event.wait()
        self._logger.info("UDP publisher worker ready to begin processing")
        
        q = self._message_queue
        has_task_done = hasattr(q, 'task_done') # Compatiblity with other queues
        
        while True:
            # Get message from queue based on blocking configuration
            message = self._dequeue_message()
            
            # Handle case where no message was retrieved
            if message is None:
                continue
            # Check for sentinel (shutdown signal)
            if message is self._sentinel:
                if has_task_done:
                    q.task_done()
                self._logger.info(
                    "UDP publisher worker received shutdown signal"
                )
                break
            
            # Process regular message - send via UDP
            try:
                self._sock.sendto(message, (self.udp_ip, self.udp_port))
                self._logger.debug(
                    f"Sent UDP message ({len(message)} bytes)"
                )
            except Exception as e:
                self._logger.error(f"Failed to send UDP message: {e}")
            
            # Mark task as done
            if has_task_done:
                q.task_done()
        
        self._logger.info("UDP publisher worker stopped")

    #########################
    ### Exposed functions ###
    #########################
    
    def cleanup(self) -> None:
        """
        Cleanup drone resources and stop background threads.
        
        Args:
            None
            
        Returns:
            None
        """
        self._logger.info("Cleaning up drone resources...")
        self._shutdown_event.set()
        self._stop_threads()
        if hasattr(self, '_sock') and self._sock:
            self._sock.close()
        self._start_event.clear()
        self._shutdown_event.clear()
        self._logger.info("Drone cleanup complete")
    
    def run(self) -> None:
        """
        Run the drone in the appropriate mode based on debug settings.
        This method blocks until interrupted.
        
        Args:
            None
            
        Returns:
            None
        """
        if self.debug_camera:
            self._logger.info("Running in camera-control-only mode")
        elif self.debug_detect:
            self._logger.info("Running in detection-only mode with vehicle")
        elif self.debug_detect_no_vech:
            self._logger.info("Running in detection-only mode without vehicle")
        elif self.debug_goto_waypoints:
            self._logger.info("Running in goto-waypoints-only mode")
        else:
            self._logger.info("Running in default mode")
        
        # Wait for shutdown event while worker threads process in background
        self._shutdown_event.wait()

    def start(self, show_preview: bool = True) -> None:
        """
        Start the drone system.
        
        Args:
            show_preview: Whether to show camera preview (default True)
            
        Returns:
            None
        """
        config = self._picam2.create_preview_configuration(
            controls={"FrameRate": self._intrinsics.inference_rate}, 
            buffer_count=self.buffer_count,
            transform=libcamera.Transform(
                hflip=self.hflip, 
                vflip=self.vflip
            )
        )
        
        self._imx500.show_network_fw_progress_bar()
        self._picam2.start(config, show_preview=show_preview)
        
        if self._intrinsics.preserve_aspect_ratio:
            self._imx500.set_auto_aspect_ratio()
            
        self._picam2.pre_callback = self._draw_detections

        self._pwm.start(5) # neutral (0 degrees)
        self.base_angle = 35  # servo neutral position (camera tilted)

        # Start all required threads
        self._start_threads()
        
        # Signal all threads to begin processing
        self._logger.info("Signaling threads to begin processing")
        self._start_event.set()

def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for the drone system.
    
    Args:
        None
        
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Simple Payload Drone")
    
    # Debug
    parser.add_argument(
        "--debug-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Runs camera pitch control only with vehicle connection"
    )
    parser.add_argument(
        "--debug-detect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Runs detection only with vehicle connection for location"
    )
    parser.add_argument(
        "--debug-detect-no-vech",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Runs detection only without vehicle connection (no location)"
    )
    parser.add_argument(
        "--debug-goto-waypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Runs go to waypoints only with vehicle connection"
    )
    parser.add_argument(
        "--show-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show preview of detection"
    )
    
    # Directories
    parser.add_argument(
        "--model-dir", 
        default="/home/useradmin/imx500-models/", 
        help="Path to directory containing models",
        type=str
    )
    parser.add_argument(
        "--labels-dir",
        default="/home/useradmin/simple_payload_autonomy_nr/labels",
        help="Path to directory containing labels",
        type=str
    )
    parser.add_argument(
        "--wpt-json-dir",
        default="./waypoints/",
        help="Path to directory containing waypoint JSON files",
        type=str
    )
    
    # Logging
    parser.add_argument(
        "--log-dir",
        default="./logs/target_detection/",
        help="Directory to save log files with timestamps",
        type=str
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Logging level",
        type=str
    )
    
    # Model
    parser.add_argument(
        "--model-type",
        choices=[
            "yolo11n", 
            "yolov8n",
            "efficientdet_lite0",
            "nanodet_plus",
            "nanodet_plus_pp",
            "ssd_mobilenetv2_fpnlite",
        ],
        default="yolo8n",
        help="Type of models to be used", 
        type=str
    )
    
    # Model intrinsics
    parser.add_argument(
        "--bbox-normalization", 
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize bbox"
    )
    parser.add_argument(
        "--bbox-order", 
        choices=["yx", "xy"], 
        default="xy", 
        help="bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
    )
    parser.add_argument(
        "--fps",
        default=30,
        help="Frames per second",
        type=int
    )
    parser.add_argument(
        "--ignore-dash-labels", 
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Remove '-' labels"
    )
    parser.add_argument(
        "--inference-rate",
        default=30.0,
        help="Inference rate",
        type=float
    )
    parser.add_argument(
        "--labels",
        choices=["coco"],
        default="coco",
        help="Labels name",
        type=str,
    )
    parser.add_argument(
        "--postprocess", 
        choices=["", "nanodet"],
        default="", 
        help="Run post process of type"
    )
    parser.add_argument(
        "-r", 
        "--preserve-aspect-ratio", 
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the pixel aspect ratio of the input tensor"
    )
    parser.add_argument(
        "--softmax",
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Softmax"
    )
    parser.add_argument(
        "--use-default-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=False, 
        help="Use default intrinsics parameters"
    )
    
    # Camera parameters
    parser.add_argument(
        "--buffer-count",
        default=12,
        help="Frames captured and stored before processing",
        type=int,
    )

    # Comms parameters
    parser.add_argument(
        "--rpi-baud-rate",
        default=57600,
        help="Raspberry Pi to FCU Baud rate",
        type=int
    )
    parser.add_argument(
        "--rpi-serial-port",
        default="/dev/ttyAMA0",
        help="Raspberry Pi Serial Port",
        type=str
    )
    parser.add_argument(
        "--udp-image-quality",
        default=30,
        help=(
            "JPEG quality for UDP image transmission" 
            "(1-100, higher = better quality but larger size)"
        ),
        type=int
    )
    parser.add_argument(
        "--udp-ip",
        default="10.5.0.1",
        help="UDP IP Address",
        type=str
    )
    parser.add_argument(
        "--udp-port",
        default=5602,
        help="UDP Port",
        type=int
    )
    parser.add_argument(
        "--udp-pub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pub results",
    )
    parser.add_argument(
        "--udp-queue-block",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether UDP queue operations should block when full/empty"
    )
    parser.add_argument(
        "--udp-queue-maxsize",
        default=0,
        help=(
            "Maximum size of UDP message queue "
            "(0 = unlimited, higher values may use more memory)"
        ),
        type=int
    )
    parser.add_argument(
        "--udp-queue-timeout",
        default=None,
        help=(
            "Timeout in seconds for UDP message queue operations "
            "(None = no timeout, only used when blocking is enabled)"
        ),
        type=lambda x: None if x.lower() == 'none' else float(x)
    )

    # Detection parameters
    parser.add_argument(
        "--detect-classes",
        default=[],
        nargs="+",
        help="List of classes (int) to detect",
        type=int
    )
    parser.add_argument(
        "--hflip",
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Whether to flip input image over horizontal plane"
    )
    parser.add_argument(
        "--iou", 
        default=0.65, 
        help="IOU threshold",
        type=float,
    )
    parser.add_argument(
        "--max-detections", 
        default=10, 
        help="Max detections",
        type=int,
    )
    parser.add_argument(
        "--threshold",
        default=0.55,
        help="Detection threshold",
        type=float,
    )
    parser.add_argument(
        "--vflip",
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Whether to flip input image over vertical plane"
    )

    # Drone parameters
    parser.add_argument(
        "--drone-id",
        default="drone_0",
        help="Drone identification string",
        type=str
    )

    # Waypoint parameters
    parser.add_argument(
        "--wpt-json-filename",
        default="waypoints.json",
        help="Filename of the waypoint JSON file",
        type=str
    )
    parser.add_argument(
        "--wpt-arrival-threshold",
        default=0.01,
        help="Waypoint arrival threshold as fraction of initial distance",
        type=float
    )
    
    return parser.parse_args()

def main() -> None:
    """
    Main entry point for the drone application.
    
    Args:
        None
        
    Returns:
        None
    """
    args = get_args()
    drone = SimplePayloadDrone(args)
    
    try:
        drone.start(show_preview=args.show_preview)
        drone.run()
    except KeyboardInterrupt:
        drone.logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        drone.logger.error(f"Application error: {e}")
    finally:
        drone.cleanup()

if __name__ == "__main__":
    
    main()
