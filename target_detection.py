import argparse
import cv2
import numpy as np
import libcamera
import logging
import os
import sys
import time
import socket
import serial

from datetime import datetime
from dronekit import connect, VehicleMode, LocationGlobalRelative
from dataclasses import dataclass
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection
)

@dataclass
class Detection:
    category: int
    conf: float
    box: tuple
        
class IMX500Detector:
    def __init__(self, args: argparse.Namespace):
        
        # Logger
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging._nameToLevel.get(args.log_level))
        # Add StreamHandler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(name)s - %(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        
        # Get model path
        self.model_type = args.model_type
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
        if args.labels == "coco":
            labels_name = "coco_labels.txt"
        labels_path = os.path.join(args.labels_dir, labels_name)
        
        # Initialize IMX500
        self.imx500 = IMX500(model_path)
        
        # Create intrinsics
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self._logger.info("No intrinsics provided")
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"
        if self.intrinsics.task != "object detection":
            raise ValueError(
                f"Model type {self.model_type} cannot be used for OD"
            )
        # Update intrinsics
        if args.use_default_intrinsics:
            self.intrinsics.update_with_defaults()
        else:
            self.intrinsics.bbox_normalization = args.bbox_normalization
            self.intrinsics.bbox_order = args.bbox_order
            self.intrinsics.fps = args.fps
            self.intrinsics.ignore_dash_labels = args.ignore_dash_labels
            self.intrinsics.inference_rate = args.inference_rate
            with open(labels_path, "r") as f:
                self.intrinsics.labels = f.read().splitlines()
            self.intrinsics.postprocess = args.postprocess
            self.intrinsics.preserve_aspect_ratio = \
                args.preserve_aspect_ratio
            self.intrinsics.softmax = args.softmax
        
        # Initialize camera
        self.picam2 = Picamera2(self.imx500.camera_num)
        
        # Camera parameters
        self.buffer_count = args.buffer_count
        
        # Detection parameters
        self.hflip = args.hflip
        self.iou = args.iou
        self.max_detections = args.max_detections
        self.threshold = args.threshold
        self.vflip = args.vflip
        
        # Dash removed labels
        if self.intrinsics.ignore_dash_labels:
            self.dash_rmv_labels = [
                label for label in self.intrinsics.labels \
                if label and label != "-"
            ]

        # Store detections
        self.last_detections: list[Detection] = []
        self.last_results: list[Detection] = None
        
        # Connection parameters
        self.rpi_baud_rate = args.rpi_baud_rate
        self.rpi_serial_port = args.rpi_serial_port
        self.udp_ip = args.udp_ip
        self.udp_port = args.udp_port
        
        # Setup comms
        if not args.debug:
            self.sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.vehicle=connect(
                self.rpi_serial_port, 
                wait_ready=True, 
                baud=self.rpi_baud_rate
            )
        
    @property
    def logger(self):
        """ Get logger"""
        return self._logger
    
    def start(self, show_preview=True):
        """Start the detector"""
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.intrinsics.inference_rate}, 
            buffer_count=self.buffer_count,
            transform=libcamera.Transform(
                hflip=self.hflip, 
                vflip=self.vflip
            )
        )
        
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(config, show_preview=show_preview)
        
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()
            
        self.picam2.pre_callback = self.draw_detections
    
    def parse_detections(self, metadata: dict):
        """
        Parse the output tensor into a number of detected objects, 
        scaled to the ISP output.
        """
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self.imx500.get_input_size()
        if np_outputs is None:
            return self.last_detections
        if self.intrinsics.postprocess == "nanodet":
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
            if self.intrinsics.bbox_normalization:
                # Assumes input_h == input_w
                boxes = boxes / input_h

            if self.intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)
        
        self.last_detections = [
            Detection(
                int(category),
                score,
                self.imx500.convert_inference_coords(
                    box, 
                    metadata,
                    self.picam2
                )
            )
            for box, score, category in zip(boxes, scores, classes)
            if score > self.threshold
        ]
        return self.last_detections

    def draw_detections(self, request, stream="main"):
        """Draw the detections for this request onto the ISP output."""
        if self.last_results is None:
            return
        if self.intrinsics.ignore_dash_labels:
            labels = self.dash_rmv_labels
        else:
            labels = self.intrinsics.labels
            
        with MappedArray(request, stream) as m:
            for d in self.last_results:
                x, y, w, h = d.box
                label = f"{labels[d.category]} ({d.conf:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    1
                )
                text_x = x + 5
                text_y = y + 15

                # Create a copy of array to draw background with opacity
                overlay = m.array.copy()

                # Draw the background rectangle on the overlay
                cv2.rectangle(overlay,
                              (text_x, text_y - text_height),
                              (text_x + text_width, text_y + baseline),
                              (255, 255, 255), # White background color
                              cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(
                    overlay, 
                    alpha, 
                    m.array, 
                    1 - alpha, 
                    0, 
                    m.array
                )

                # Draw text on top of the background
                cv2.putText(
                    m.array, 
                    label, 
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    1
                )

                # Draw detection box
                cv2.rectangle(
                    m.array, 
                    (x, y), 
                    (x + w, y + h), 
                    (0, 255, 0, 0), 
                    thickness=2
                )

            if self.intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
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
    
    def detect(self):
        """ Run detection """
        while True:
            # Get the latest detections
            self.last_results = self.parse_detections(
                self.picam2.capture_metadata()
            )
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
    
    def detect_and_pub(self, classes: list[int]):
        """ Run detection and publish info for interested classes """
        while True:
            # Get the latest detections
            self.last_results = self.parse_detections(
                self.picam2.capture_metadata()
            )
            
            for detection in self.last_results:
                detected_class = detection.category
                confidence = detection.conf
        
                # Print when a target is detected with high confidence
                if detected_class in classes:
                    location=self.vehicle.location.global_frame
                    message=(
                        f"{datetime.now()}: "
                        f"{self.intrinsics.labels[detected_class]} "
                        f"detected @ {location} with "
                        f"{confidence:.2f} confidence"
                    )
                    self._logger.debug(message)
                    data=message.encode('utf-8')
                    self.sock.sendto(data, (self.udp_ip, self.udp_port))
    
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
    
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
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debugging mode. Runs detect()"
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
    
    # Detection parameters``
    parser.add_argument(
        "--detect-classes",
        default=0, 
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
    
    return parser.parse_args()

def main():
    
    try:
        args = get_args()
        detector = IMX500Detector(args)
        detector.start(show_preview=args.show_preview)
        if args.debug:
            detector.detect()
        else:
            detector.detect_and_pub(args.detect_classes)
    except KeyboardInterrupt as e:
        detector.logger.info(f"{e}")

if __name__ == "__main__":
    
    main()
