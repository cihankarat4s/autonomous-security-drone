import csv
from pathlib import Path


class FrameLog:
    """One CSV row per processed frame, for comparing models in flight.

    mission_seq is the last MISSION_ITEM_REACHED, so rows can be grouped by
    mission leg (e.g. the hover over the protected area).
    """

    FIELDS = ["t_s", "mission_seq", "alt_m", "persons", "best_conf", "inference_ms"]

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def write(self, t_s, mission_seq, alt_m, detections, inference_s):
        best = max((d.confidence for d in detections), default=0.0)
        self._writer.writerow([
            f"{t_s:.2f}",
            "" if mission_seq is None else mission_seq,
            "" if alt_m is None else f"{alt_m:.1f}",
            len(detections),
            f"{best:.3f}",
            f"{inference_s * 1000:.1f}",
        ])
        self._file.flush()  # keep the rows if the companion is killed

    def close(self):
        self._file.close()
