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
import threading
import socket

from datetime import datetime
from dronekit import connect, VehicleMode, LocationGlobalRelative
from dataclasses import dataclass
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection
)
from rpi_hardware_pwm import HardwarePWM

@dataclass
class Detection:
    category: int
    conf: float
    box: tuple[int, int, int, int]
        
class IMX500Detector:
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
        
        # Threads
        
        # Camera pitch worker thread
        self._camera_pitch_thread = None
        self._camera_pitch_active = False
        # Detection worker thread
        self._detection_thread = None
        self._detection_active = False
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
        self._pwm = HardwarePWM(0, 50)  # 50 Hz for servo
        # Setup UDP socket
        if args.udp_pub:
            self._sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Store detections with thread safety
        self._detections_lock = threading.Lock()
        self._last_detections: list[Detection] = []
    
    #########################
    ### Getters / Setters ###
    #########################

    @property
    def logger(self):
        """ Get logger"""
        return self._logger
    
    ################################
    ### Private Helper Functions ###
    ################################
    
    def _draw_detections(self, request, stream="main"):
        """Draw the detections for this request onto the ISP output."""
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
    
    def _encode_detection_message(
            self, 
            detections: list[Detection], 
            image: np.ndarray,
            location: dict | None = None
        ) -> bytes:
        """
        Encode detections and image into bytes for UDP transmission.
        """
        # Encode image to JPEG bytes with configurable quality
        _, img_encoded = cv2.imencode(
            '.jpg', 
            image, 
            [cv2.IMWRITE_JPEG_QUALITY, self.udp_image_quality]
        )
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        # Convert detections to serializable format
        detection_data = []
        
        for d in detections:
            detection_data.append({
                'box': d.box, # tuple -> JSON array automatically
                'category': d.category, # int
                'confidence': d.conf, # float
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
        return json.dumps(message).encode('utf-8')
    
    def _parse_detections(self, request):
        """
        Parse the output tensor into a number of detected objects, scaled to the
        ISP output.
        
        Args:
            request: Picamera2 request object with both metadata and image data
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
            from picamera2.devices.imx500.postprocess import scale_boxes
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
                int(category),
                score,
                self._imx500.convert_inference_coords(
                    box, 
                    metadata,
                    self._picam2
                )
            )
            for box, score, category in zip(boxes, scores, classes)
            if score > self.threshold
        ]
        
        return new_detections, request.make_array("main")

    ################################
    ### Drawing helper functions ###
    ################################

    def _draw_detections_on_image(
            self,
            detections: list[Detection],
            image: np.ndarray,
        ) -> np.ndarray:
        """
        Helper function to draw detection bounding boxes and labels on an image.
        Returns a copy of the image with detections drawn.
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
        Modifies the image in-place.
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

    ######################################
    ### Message queue helper functions ###
    ######################################

    def _dequeue_message(self):
        """
        Helper function to dequeue a message with proper error handling.
        
        Returns:
            Message from queue or None if error/timeout
        """
        try:
            message = self._message_queue.get(
                block=self.udp_queue_block, 
                timeout=self.udp_queue_timeout
            )
            self._logger.debug("Dequeued message from UDP queue")
            return message
        except queue.Empty:
            self._logger.warning(
                "No message available in UDP queue (timeout/empty)"
            )
            return None
        except Exception as e:
            self._logger.error(f"Failed to dequeue message: {e}")
            return None

    def _enqueue_message(self, message):
        """
        Helper function to enqueue a message with proper error handling.
        
        Args:
            message: Message to enqueue
        """
        try:
            self._message_queue.put(
                message, 
                block=self.udp_queue_block, 
                timeout=self.udp_queue_timeout
            )
            self._logger.debug("Queued detection message for UDP publishing")
        except queue.Full:
            self._logger.warning(
                "Message queue is full, dropping detection message"
            )
        except Exception as e:
            self._logger.error(f"Failed to queue detection message: {e}")
    
    def _enqueue_sentinel(self):
        """
        Enqueue the sentinel object to signal shutdown to UDP worker thread.
        """
        # Always block indefinitely for sentinel to ensure shutdown happens
        self._message_queue.put(self._sentinel, block=True, timeout=None)

    ################################
    ### Vehicle helper functions ###
    ################################
    def _get_vehicle_location(self):
        """
        Get current vehicle location if vehicle is connected.
        
        Returns:
            dict or None: Location data with lat/lng or None if no vehicle
        """
        if self._vehicle is None:
            return {
                'latitude': None,
                'longitude': None,
                'altitude': None
            }
            
        try:
            location = self._vehicle.location.global_frame
            return {
                'latitude': location.lat,
                'longitude': location.lon,
                'altitude': location.alt
            }
        except Exception as e:
            self._logger.warning(f"Failed to get vehicle location: {e}")
            return {
                'latitude': None,
                'longitude': None,
                'altitude': None
            }

    def _get_vehicle_pitch(self):
        """
        Get current vehicle pitch if vehicle is connected.
        
        Returns:
            float or None: Pitch or None if no vehicle
        """
        if self._vehicle is None:
            return None
        try:
            return self._vehicle.attitude.pitch
        except Exception as e:
            self._logger.warning(f"Failed to get vehicle pitch: {e}")
            return None

    ################################
    ### Thread handler functions ###
    ################################

    def _start_camera_pitch_worker(self):
        """
        Start the camera pitch worker thread.
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

    def _stop_camera_pitch_worker(self):
        """
        Stop the camera pitch worker thread gracefully.
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

    def _start_detection_worker(self):
        """
        Start the detection worker thread.
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

    def _stop_detection_worker(self):
        """
        Stop the detection worker thread gracefully.
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

    def _start_udp_publisher(self):
        """
        Start the UDP publisher thread.
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
    
    def _stop_udp_publisher(self):
        """
        Stop the UDP publisher thread gracefully.
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
    
    def _start_threads(self):
        """
        Start all required threads based on operation mode.
        """
        self._logger.info("Starting threads...")
        
        if self.debug_camera:
            self._start_camera_pitch_worker()
        elif self.debug_detect or self.debug_detect_no_vech:
            self._start_detection_worker()
        else:
            self._start_camera_pitch_worker()
            self._start_detection_worker()
        
        if self.udp_pub:
            self._start_udp_publisher()
        
        self._logger.info("All threads started successfully")

    def _stop_threads(self):
        """
        Stop all running threads gracefully.
        """
        self._logger.info("Stopping threads...")
        
        # Stop camera pitch worker if running
        if self._camera_pitch_thread is not None:
            self._stop_camera_pitch_worker()
        
        # Stop detection worker if running
        if self._detection_thread is not None:
            self._stop_detection_worker()
        
        # Stop UDP publisher if running
        if self._udp_thread is not None:
            self._stop_udp_publisher()
        
        self._logger.info("All threads stopped")

    ###############################
    ### Thread worker functions ###
    ###############################

    def _camera_pitch_worker(self):
        """
        Worker thread function that handles camera pitch correction with servo.
        """
        self._logger.info("Camera pitch worker started")
        
        # Wait for synchronized start signal
        self._start_event.wait()
        self._logger.info("Camera pitch worker ready to begin processing")
        
        while self._camera_pitch_active and not self._shutdown_event.is_set():
            try:
                pitch = self._get_vehicle_pitch()
                if pitch is None:
                    self._logger.warning(
                        "No pitched obtained. Skipping servo correction."
                    )
                    continue
                else:
                    pitch = math.degrees(pitch))
                    self._logger.debug(f"Pitch: {pitch:.2f} degrees")
                        
                # Compensate
                servo_angle = self.base_angle - pitch  
                # Limit servo movement
                servo_angle = max(min(servo_angle, 90), 0)
                # Convert angle to duty cycle
                duty_cycle = 5 + (servo_angle / 90) * 5  
                self._logger.debug(
                    f"Servo angle: {servo_angle:.1f}°, Duty: {duty_cycle:.2f}%"
                )
                self._pwm.change_duty_cycle(duty_cycle)
                # time.sleep(0.02)  # 20 ms update
            except Exception as e:
                self._logger.error(f"Camera pitch worker error: {e}")
        
        self._logger.info("Camera pitch worker stopped")

    def _detection_worker(self):
        """
        Worker thread function that continuously runs detection processing.
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
                if current_detections and current_image:
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
                        lat = vehicle_location['latitude']
                        lng = vehicle_location['longitude'] 
                        alt = vehicle_location['altitude']
                        msg = (
                            f"{datetime.now().isoformat()}: "
                            f"{self._intrinsics.labels[d.category]} "
                            f"detected with {d.conf:.2f} confidence "
                            f"at lat:{lat}, lng:{lng}, alt:{alt}"
                        )
                        self._logger.debug(msg)
                
                # Send to message queue for UDP publishing if enabled
                if self.udp_pub and relevant_detections and current_image:
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
            except Exception as e:
                self._logger.error(f"Detection worker error: {e}")
        
        self._logger.info("Detection worker stopped")

    def _udp_publisher_worker(self):
        """
        Worker thread function to process message queue and send UDP messages.
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
    
    def cleanup(self):
        """
        Cleanup resources and stop background threads.
        """
        self._logger.info("Cleaning up detector resources...")
        self._shutdown_event.set()
        self._stop_threads()
        self._start_event.clear()
        self._shutdown_event.clear()
        self._logger.info("Detector cleanup complete")
    
    def run(self):
        """
        Run the detector in the appropriate mode based on debug settings.
        This method blocks until interrupted.
        """
        if self.debug_camera:
            self._logger.info("Running in camera-control-only mode")
        elif self.debug_detect:
            self._logger.info("Running in detection-only mode with vehicle")
        elif self.debug_detect_no_vech:
            self._logger.info("Running in detection-only mode without vehicle")
        else:
            self._logger.info("Running in default mode")
        
        # Wait for shutdown event while worker threads process in background
        self._shutdown_event.wait()

    def start(self, show_preview=True):
        """Start the detector"""
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

def get_args():
    parser = argparse.ArgumentParser()
    
    # Camera
    parser.add_argument(
        "--buffer-count",
        default=12,
        help="Frames captured and stored before processing",
        type=int,
    )
    
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
        help="Runs detection with vehicle connection for location"
    )
    parser.add_argument(
        "--debug-detect-no-vech",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Runs detection without vehicle connection (no location)"
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
        default=85,
        help=(
            "JPEG quality for UDP image transmission" 
            "(1-100, higher = better quality but larger size)"
        ),
        type=int
    )
    parser.add_argument(
        "--udp-ip",
        default="192.168.0.109",
        help="UDP IP Address",
        type=str
    )
    parser.add_argument(
        "--udp-port",
        default=14550,
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
    
    return parser.parse_args()

def main():
    args = get_args()
    detector = IMX500Detector(args)
    
    try:
        detector.start(show_preview=args.show_preview)
        detector.run()
    except KeyboardInterrupt:
        detector.logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        detector.logger.error(f"Application error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    
    main()
