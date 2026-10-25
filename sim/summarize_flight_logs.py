"""Compare per-frame detection logs (companion --frame-log) of several flights.

The companion only runs detection while it patrols the protected area, so by
default every logged frame is scored. --seq scores one mission leg, e.g.
--seq 2 for the hover of the earlier --hover layout (after WP2 is reached and
before the LOITER_TIME item reports).

    python sim/summarize_flight_logs.py logs/flight_coco.csv logs/flight_visdrone.csv
"""
import argparse
import csv
from pathlib import Path

FIELDS = ["log", "frames", "detection_rate", "mean_conf", "min_conf",
          "longest_gap_s", "mean_inference_ms"]


def summarize(rows, seq=None):
    """Detection statistics of all rows, or of those whose mission_seq is seq."""
    leg = [r for r in rows if seq is None or r["mission_seq"] == str(seq)]
    if not leg:
        return None
    hits = [r for r in leg if int(r["persons"]) > 0]
    confs = [float(r["best_conf"]) for r in hits]

    # Longest stretch without a detection, from the last hit (or leg start)
    # to the next hit (or leg end).
    longest_gap = 0.0
    gap_start = float(leg[0]["t_s"])
    for r in leg:
        t = float(r["t_s"])
        if int(r["persons"]) > 0:
            longest_gap = max(longest_gap, t - gap_start)
            gap_start = t
    longest_gap = max(longest_gap, float(leg[-1]["t_s"]) - gap_start)

    return {
        "frames": len(leg),
        "detection_rate": round(len(hits) / len(leg), 3),
        "mean_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "min_conf": round(min(confs), 3) if confs else 0.0,
        "longest_gap_s": round(longest_gap, 2),
        "mean_inference_ms": round(sum(float(r["inference_ms"]) for r in leg) / len(leg), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--seq", type=int, help="only score this mission leg")
    parser.add_argument("--out", default="docs/results/gazebo_flight.csv")
    args = parser.parse_args()

    results = []
    for path in args.logs:
        with path.open(newline="", encoding="utf-8") as f:
            summary = summarize(list(csv.DictReader(f)), args.seq)
        if summary is None:
            print(f"{path}: no frames to score")
            continue
        results.append({"log": path.name, **summary})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(" | ".join(FIELDS))
    for r in results:
        print(" | ".join(str(r[k]) for k in FIELDS))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
