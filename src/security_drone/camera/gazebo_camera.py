"""Camera frames from the Gazebo simulation.

Any camera source (this one, a PiCamera later) exposes ``read()`` that returns
the latest frame as a BGR uint8 array (OpenCV convention) or None.
"""
import threading

import numpy as np

# gz.msgs.PixelFormatType values, hard-coded so the conversion needs no gz import.
RGB_INT8 = 3
BGR_INT8 = 8


def image_to_bgr(msg):
    """Convert a gz.msgs.Image (or anything with the same fields) to a BGR array."""
    if msg.pixel_format_type not in (RGB_INT8, BGR_INT8):
        raise ValueError(f"Unsupported pixel format {msg.pixel_format_type}")
    # Rows may be padded: step is the row length in bytes.
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.pixel_format_type == RGB_INT8:
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


class GazeboCamera:
    """Subscribes to a gz camera topic and keeps the latest frame."""

    def __init__(self, topic):
        self.topic = topic
        self._lock = threading.Lock()
        self._frame = None
        self._node = None

    def start(self):
        # Imported here so the module (and its tests) work without the gz bindings.
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node

        self._node = Node()
        if not self._node.subscribe(Image, self.topic, self._on_image):
            raise RuntimeError(f"Could not subscribe to {self.topic}")

    def _on_image(self, msg):
        frame = image_to_bgr(msg)
        with self._lock:
            self._frame = frame

    def read(self):
        """Latest frame, or None if nothing has arrived yet."""
        with self._lock:
            return self._frame
