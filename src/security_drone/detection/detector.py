from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def merge_overlapping(detections, overlap=0.5):
    """Keep one box per person: drop boxes that mostly cover a kept box.

    YOLO26 has no NMS step and can put several boxes on one person (a small
    box on the body inside larger ones with the shadow). Their IoU is low, so
    plain NMS would keep them; instead a box is dropped when its intersection
    with an already kept, more confident box is more than `overlap` of the
    smaller of the two. Two people side by side overlap far less than that.
    """
    kept = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        for k in kept:
            iw = min(det.x2, k.x2) - max(det.x1, k.x1)
            ih = min(det.y2, k.y2) - max(det.y1, k.y1)
            if iw > 0 and ih > 0 and iw * ih > overlap * min(det.area, k.area):
                break
        else:
            kept.append(det)
    return kept


class PersonDetector:
    """YOLO (Ultralytics) restricted to the person class(es)."""

    def __init__(self, model_path, confidence, classes=(0,), image_size=640, device=None):
        # Imported here so tests and tools that never detect skip loading torch.
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        # COCO person is class 0; a VisDrone model would use its own ids.
        self.classes = list(classes)
        self.image_size = image_size
        self.device = device

    def warm_up(self, height=480, width=640):
        """Run one blank frame so CUDA setup doesn't delay the first real one."""
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))

    def detect(self, frame):
        result = self.model.predict(
            frame,
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        return merge_overlapping([
            Detection(*box.xyxy[0].tolist(), float(box.conf[0]))
            for box in result.boxes
        ])


def draw_detections(frame, detections):
    """Copy of frame with a box and confidence for each detection."""
    out = frame.copy()
    for det in detections:
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
        cv2.putText(out, f"person {det.confidence:.2f}", (p1[0], max(p1[1] - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out
