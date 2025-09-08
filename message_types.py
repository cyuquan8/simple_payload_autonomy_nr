"""
Shared message types and enums for drone-ground station communication.

This module defines enums and constants used for structured communication
between the drone and ground station systems.
"""

from enum import StrEnum, auto

class MessageType(StrEnum):
    """Message types for drone-ground station communication."""
    DETECTION = auto()
    TELEMETRY = auto()
    TAKEOFF = auto()
    RTL = auto()
    WAYPOINT = auto()
    UNKNOWN = auto()

class TakeoffStatus(StrEnum):
    """Takeoff operation status values."""
    STARTED = auto()
    COMPLETED = auto()
    ABORTED = auto()

class RTLStatus(StrEnum):
    """Return to Launch operation status values."""
    STARTED = auto()
    COMPLETED = auto()
    ABORTED = auto()

class WaypointStatus(StrEnum):
    """Waypoint navigation status values."""
    STARTED = auto()
    REACHED = auto()
    COMPLETED = auto()
    ABORTED = auto()