"""Replay recorded flight frames and compare alarm rules.

Frames come from the companion's --record-dir (names carry the time since
detection started). For each rule the alarm state machine is run exactly as
in the companion (ThreatAlarm, same clear and repeat times), and the script
reports how often the alarm went off, how long it stayed on and when it
first went off:

    frame        any box in the frame counts (the companion until now)
    track K in W a track (CentroidTracker, camera shift compensated) must be
                 seen in K frames within W seconds

On a person-free flight every alarm is false; on a flight with people the
cost of a rule is a later or shorter alarm.

    python training/replay_alarm.py logs/nb_rec_empty_frames logs/nb_rec_people_frames \
        --model yolo26n_visdrone_person_neg.pt --save-dir logs/replay
"""
import argparse
import re
from pathlib import Path

import cv2
from ultralytics import YOLO

from security_drone.alert.alarm import AlarmEvent, ThreatAlarm
from security_drone.detection.confirm import CentroidTracker, TrackConfirmer, camera_shift
from security_drone.detection.detector import Detection, merge_overlapping

CLEAR_AFTER_S, NOTIFY_EVERY_S = 3.0, 10.0      # config/default.yaml alert


def frame_time(path):
    return float(re.search(r"_t([\d.]+)_", path.name).group(1))


def run_alarm(counts, times):
    """Alarm activations, seconds with the alarm on, and first activation time."""
    alarm = ThreatAlarm(CLEAR_AFTER_S, NOTIFY_EVERY_S)
    activations, on_s, first = 0, 0.0, None
    prev = times[0]
    for count, t in zip(counts, times):
        if alarm.active:
            on_s += t - prev
        if alarm.update(count, t) is AlarmEvent.ACTIVATED:
            activations += 1
            first = t if first is None else first
        prev = t
    return activations, on_s, first


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--model", default="yolo26n_visdrone_person_neg.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--rules", nargs="+", default=["2:1", "3:1", "3:2", "4:3"],
                        help="track rules as hits:window_seconds")
    parser.add_argument("--gate", type=float, default=60.0, help="tracker match radius, px")
    parser.add_argument("--save-dir", type=Path, help="annotated frames of confirmed tracks")
    args = parser.parse_args()

    print("flight | rule | alarms | alarm on (s) | first alarm (s) | frames with a box")
    for folder in args.folders:
        frames = sorted(folder.glob("*.jpg"))
        times = [frame_time(f) for f in frames]
        detector = YOLO(args.model)
        tracker = CentroidTracker(max_px=args.gate)
        boxes, per_frame = [], []          # per frame: box count; [(track id, conf, xyxy), ...]
        prev = None
        for f, t in zip(frames, times):
            img = cv2.imread(str(f))
            # The companion's own rule: predict, nested boxes merged.
            r = detector.predict(img, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)[0]
            dets = merge_overlapping([Detection(*b, c) for b, c in
                                      zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
            boxes.append(len(dets))
            shift = camera_shift(prev, img) if prev is not None else (0.0, 0.0)
            prev = img
            ids = tracker.update([((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in dets], t, shift)
            per_frame.append([(tid, d.confidence, (d.x1, d.y1, d.x2, d.y2)) for tid, d in zip(ids, dets)])

        n_box = sum(1 for b in boxes if b)
        a, on, first = run_alarm(boxes, times)
        print(f"{folder.name} | frame | {a} | {on:.1f} | {first} | {n_box}/{len(frames)}")
        rules = [(int(r.split(":")[0]), float(r.split(":")[1])) for r in args.rules]
        for k, w in rules:
            confirmer = TrackConfirmer(min_hits=k, window_s=w)
            counts = []
            for (dets, t, f) in zip(per_frame, times, frames):
                ok = confirmer.update([tid for tid, _, _ in dets if tid is not None], t)
                counts.append(len(ok))
                if ok and args.save_dir and (k, w) == rules[len(rules) // 2]:
                    out = args.save_dir / folder.name
                    out.mkdir(parents=True, exist_ok=True)
                    img = cv2.imread(str(f))
                    for tid, conf, (x1, y1, x2, y2) in dets:
                        if tid in ok:
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(img, f"id{tid} {conf:.2f}", (int(x1), max(int(y1) - 4, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    cv2.imwrite(str(out / f.name), img)
            a, on, first = run_alarm(counts, times)
            print(f"{folder.name} | track {k} in {w:g} s | {a} | {on:.1f} | {first} | "
                  f"{sum(1 for c in counts if c)}/{len(frames)} confirmed")


if __name__ == "__main__":
    main()
