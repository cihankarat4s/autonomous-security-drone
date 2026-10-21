import threading
import time
from dataclasses import dataclass

from security_drone.detection.preprocess import preprocess


@dataclass
class DetectionResult:
    frame: object
    detections: list
    inference_s: float


class DetectionWorker(threading.Thread):
    """Runs detection on new camera frames in the background.

    Inference takes tens of milliseconds, so it runs off the main thread,
    which keeps reading MAVLink. Results go through a queue; only the main
    thread touches the MAVLink connection.
    """

    def __init__(self, camera, detector, results, equalize=True, sharpen_edges=True):
        super().__init__(daemon=True)
        self.camera = camera
        self.detector = detector
        self.results = results
        self.equalize = equalize
        self.sharpen_edges = sharpen_edges
        self._stop_event = threading.Event()

    def run(self):
        last_frame = None
        while not self._stop_event.is_set():
            frame = self.camera.read()
            # The camera keeps the latest frame; skip it if we already saw it.
            if frame is None or frame is last_frame:
                time.sleep(0.01)
                continue
            last_frame = frame
            start = time.perf_counter()
            detections = self.detector.detect(
                preprocess(frame, self.equalize, self.sharpen_edges))
            self.results.put(DetectionResult(frame, detections, time.perf_counter() - start))

    def stop(self):
        self._stop_event.set()
