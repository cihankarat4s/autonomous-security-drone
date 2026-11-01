"""Turn the frames of a recorded patrol into a demo video.

Frames come from the companion's --record-dir. The detector, TrackFilter and the
alarm run again on them with the settings in config/default.yaml, so the video
shows what those settings do. Grey boxes: every detection. Green: passed the
filter. Frames are repeated to keep the recorded time, so the video plays at
the speed the frames were processed.

    python sim/make_demo_video.py logs/demo1_frames logs/demo1.mp4
"""
import argparse
import re
from pathlib import Path

import cv2

from security_drone.alert.alarm import ThreatAlarm
from security_drone.config import load_config
from security_drone.detection.confirm import TrackFilter
from security_drone.detection.detector import PersonDetector

FPS = 15
GREY, GREEN, RED, WHITE = (170, 170, 170), (0, 230, 0), (0, 0, 220), (255, 255, 255)


def frame_info(path):
    match = re.search(r"_t([\d.]+)_seq(\d+|None)", path.name)
    return float(match[1]), match[2]


def draw(frame, raw, kept, t, seq, alarm_on):
    for d in raw:
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), GREY, 1)
    for d in kept:
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), GREEN, 2)
    banner = RED if alarm_on else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), banner, -1)
    state = f"THREAT: {len(kept)} person" if alarm_on else "Patrolling"
    cv2.putText(frame, f"{state}   t={t:5.1f} s   last waypoint {seq}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = load_config()
    det, tracking, alert = config["detection"], config["tracking"], config["alert"]
    detector = PersonDetector(det["model"], det["confidence"], det["classes"], det["image_size"], det["device"])
    track_filter = TrackFilter(tracking["min_hits"], tracking["window_s"], tracking["hold_s"], tracking["max_px"])
    alarm = ThreatAlarm(alert["clear_after_s"], alert["notify_every_s"])

    paths = sorted(args.frames.glob("*.jpg"))
    times = [frame_info(p)[0] for p in paths]
    writer, written, raw_frames, kept_frames = None, 0, 0, 0
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        t, seq = frame_info(path)
        raw = detector.detect(frame)
        kept = track_filter.update(frame, raw, t)
        alarm.update(len(kept), t)
        raw_frames += bool(raw)
        kept_frames += bool(kept)
        draw(frame, raw, kept, t, seq, alarm.active)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        end = times[i + 1] if i + 1 < len(times) else t + 1 / FPS
        target = round((end - times[0]) * FPS)
        while written < target:
            writer.write(frame)
            written += 1
    writer.release()
    span = times[-1] - times[0]
    print(f"{len(paths)} frames in {span:.0f} s ({len(paths) / max(span, 1):.1f} per second), "
          f"{raw_frames} with a box, {kept_frames} after the filter -> {args.output}")


if __name__ == "__main__":
    main()
