import math
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

SHIFT_WIDTH = 320    # frames are compared at this width; enough for a global shift
MIN_RESPONSE = 0.01  # phase correlation peak below this is not a real match


def camera_shift(prev, frame):
    """(dx, dy) in pixels by which the image content moved from prev to frame.

    Phase correlation of the two frames, downscaled: the ground fills the
    view, so the dominant shift is the camera's own motion. Rotation (turns
    at corners) is not modelled; the tracker's gate absorbs a little of it.
    """
    scale = SHIFT_WIDTH / frame.shape[1]
    a, b = (cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA).astype(np.float32) for f in (prev, frame))
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    if not response > MIN_RESPONSE:      # a blank frame gives a meaningless peak
        return 0.0, 0.0
    return dx / scale, dy / scale


class CentroidTracker:
    """Gives each detection a track id by matching box centres between frames.

    IoU trackers (ByteTrack) assume boxes overlap from frame to frame. From
    a moving drone at a few frames per second a person's box moves ~50 px
    between frames, so it never overlapped and every box got a new id.
    Here the previous centres are first moved by the camera's own shift
    between the frames (measured on the whole image), then each box takes
    the nearest unmatched track within max_px. Tracks unseen for more than
    max_age_s are dropped.
    """

    def __init__(self, max_px=60.0, max_age_s=1.5):
        self.max_px = max_px
        self.max_age_s = max_age_s
        self._tracks = {}          # id -> (x, y, last seen)
        self._next_id = 0

    def update(self, centres, now, shift=(0.0, 0.0)):
        """Centres (x, y) of one frame and the image shift since the last
        frame; returns one track id per centre."""
        dx, dy = shift
        self._tracks = {tid: (x + dx, y + dy, t) for tid, (x, y, t) in self._tracks.items()
                        if now - t <= self.max_age_s}
        pairs = sorted((math.hypot(cx - x, cy - y), i, tid)
                       for i, (cx, cy) in enumerate(centres)
                       for tid, (x, y, _) in self._tracks.items())
        ids, used = [None] * len(centres), set()
        for dist, i, tid in pairs:
            if dist > self.max_px:
                break
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, (cx, cy) in enumerate(centres):
            if ids[i] is None:
                ids[i] = self._next_id
                self._next_id += 1
            self._tracks[ids[i]] = (cx, cy, now)
        return ids

    def reset(self):
        self._tracks.clear()


class TrackConfirmer:
    """Confirms a tracked object only after it has been seen in several frames.

    A single-frame box raised the alarm before, so one flickering false
    positive (a dog, a hydrant seen for a moment) was enough. A track id now
    has to appear in at least min_hits frames within window_s before it
    counts; once confirmed it stays confirmed while it keeps being seen.
    """

    def __init__(self, min_hits=3, window_s=1.0):
        self.min_hits = min_hits
        self.window_s = window_s
        self._seen = {}            # track id -> deque of timestamps
        self._confirmed = set()

    def update(self, track_ids, now):
        """Feed the track ids of one frame; returns the confirmed ids in it."""
        for tid in track_ids:
            times = self._seen.setdefault(tid, deque())
            times.append(now)
        for tid in list(self._seen):
            times = self._seen[tid]
            while times and now - times[0] > self.window_s:
                times.popleft()
            if not times:              # not seen for a whole window: forget it
                del self._seen[tid]
                self._confirmed.discard(tid)
            elif len(times) >= self.min_hits:
                self._confirmed.add(tid)
        return {tid for tid in track_ids if tid in self._confirmed}

    def reset(self):
        self._seen.clear()
        self._confirmed.clear()


class TrackFilter:
    """Hides boxes seen only for a moment and keeps boxes lost for a moment.

    A detection is passed on once its track is confirmed (TrackConfirmer). A
    confirmed track that gets no box stays in the output at its last place,
    moved with the camera, for up to hold_s. The confirmed flag lasts as long
    as the track does, so a person who flickers is not confirmed again.
    """

    def __init__(self, min_hits=3, window_s=1.0, hold_s=0.5, max_px=60.0):
        self.tracker = CentroidTracker(max_px=max_px, max_age_s=max(hold_s, window_s))
        self.confirmer = TrackConfirmer(min_hits, window_s)
        self.hold_s = hold_s
        self._held = {}            # track id -> (detection, last seen)
        self._prev = None

    def update(self, frame, detections, now):
        """Boxes of one frame, unfiltered; returns the boxes to act on."""
        shift = camera_shift(self._prev, frame) if self._prev is not None else (0.0, 0.0)
        self._prev = frame
        centres = [((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in detections]
        ids = self.tracker.update(centres, now, shift)
        confirmed = self.confirmer.update(ids, now)
        dx, dy = shift
        self._held = {tid: (replace(d, x1=d.x1 + dx, y1=d.y1 + dy, x2=d.x2 + dx, y2=d.y2 + dy), t)
                      for tid, (d, t) in self._held.items() if now - t <= self.hold_s}
        passed = []
        for detection, tid in zip(detections, ids):
            if tid in confirmed or tid in self._held:
                self._held[tid] = (detection, now)
                passed.append(detection)
        seen = set(ids)
        passed += [d for tid, (d, _) in self._held.items() if tid not in seen]
        return passed

    def reset(self):
        self.tracker.reset()
        self.confirmer.reset()
        self._held.clear()
        self._prev = None
