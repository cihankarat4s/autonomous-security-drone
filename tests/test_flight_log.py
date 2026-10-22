import csv
import importlib.util
from pathlib import Path

from security_drone.detection.detector import Detection
from security_drone.detection.frame_log import FrameLog

_SCRIPT = Path(__file__).resolve().parents[1] / "sim" / "summarize_flight_logs.py"
_spec = importlib.util.spec_from_file_location("summarize_flight_logs", _SCRIPT)
summarize_flight_logs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_flight_logs)


def _write_log(path, frames):
    log = FrameLog(path)
    for t, seq, confs in frames:
        detections = [Detection(0, 0, 1, 1, c) for c in confs]
        log.write(t, seq, 15.0, detections, 0.04)
    log.close()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_frame_log_keeps_best_confidence(tmp_path):
    rows = _write_log(tmp_path / "f.csv", [(0.5, 2, [0.45, 0.8]), (0.6, None, [])])
    assert rows[0] == {"t_s": "0.50", "mission_seq": "2", "alt_m": "15.0",
                       "persons": "2", "best_conf": "0.800", "inference_ms": "40.0"}
    assert rows[1]["mission_seq"] == ""
    assert rows[1]["best_conf"] == "0.000"


def test_summary_scores_only_the_hover_leg(tmp_path):
    rows = _write_log(tmp_path / "f.csv", [
        (0.0, 2, [0.5]),
        (1.0, 2, []),
        (2.0, 2, []),
        (3.0, 2, [0.7]),
        (4.0, 3, []),      # after the hover: ignored
    ])
    s = summarize_flight_logs.summarize(rows, seq=2)
    assert s["frames"] == 4
    assert s["detection_rate"] == 0.5
    assert s["mean_conf"] == 0.6
    assert s["min_conf"] == 0.5
    assert s["longest_gap_s"] == 3.0


def test_gap_at_the_end_of_the_leg_counts(tmp_path):
    rows = _write_log(tmp_path / "f.csv", [(0.0, 2, [0.9]), (1.0, 2, []), (5.0, 2, [])])
    assert summarize_flight_logs.summarize(rows, seq=2)["longest_gap_s"] == 5.0


def test_no_frames_in_leg(tmp_path):
    rows = _write_log(tmp_path / "f.csv", [(0.0, 3, [0.9])])
    assert summarize_flight_logs.summarize(rows, seq=2) is None
