import numpy as np

from security_drone.detection.confirm import CentroidTracker, TrackConfirmer, TrackFilter
from security_drone.detection.detector import Detection


def test_single_frame_is_not_confirmed():
    c = TrackConfirmer(min_hits=3, window_s=1.0)
    assert c.update([7], 0.0) == set()
    assert c.update([], 0.1) == set()


def test_confirmed_after_min_hits_within_window():
    c = TrackConfirmer(min_hits=3, window_s=1.0)
    assert c.update([7], 0.0) == set()
    assert c.update([7], 0.1) == set()
    assert c.update([7], 0.2) == {7}


def test_hits_spread_wider_than_the_window_do_not_confirm():
    c = TrackConfirmer(min_hits=3, window_s=1.0)
    for t in (0.0, 0.8, 1.6, 2.4):
        assert c.update([7], t) == set()


def test_confirmed_track_stays_confirmed_while_seen():
    c = TrackConfirmer(min_hits=3, window_s=1.0)
    for t in (0.0, 0.1, 0.2):
        c.update([7], t)
    # Seen only every 0.9 s now: fewer than 3 hits per window, still confirmed.
    assert c.update([7], 1.1) == {7}
    assert c.update([7], 2.0) == {7}


def test_track_lost_for_a_window_is_forgotten():
    c = TrackConfirmer(min_hits=3, window_s=1.0)
    for t in (0.0, 0.1, 0.2):
        c.update([7], t)
    c.update([], 1.5)
    assert c.update([7], 1.6) == set()


def test_only_ids_in_the_current_frame_are_returned():
    c = TrackConfirmer(min_hits=2, window_s=1.0)
    c.update([1, 2], 0.0)
    assert c.update([1], 0.1) == {1}


def test_tracker_keeps_the_id_of_a_box_that_moved_a_little():
    tr = CentroidTracker(max_px=60)
    a = tr.update([(100, 100)], 0.0)
    assert tr.update([(140, 110)], 0.4) == a


def test_tracker_gives_a_new_id_beyond_the_gate():
    tr = CentroidTracker(max_px=60)
    a = tr.update([(100, 100)], 0.0)
    assert tr.update([(300, 100)], 0.4) != a


def test_camera_shift_is_compensated():
    tr = CentroidTracker(max_px=20)
    a = tr.update([(100, 100)], 0.0)
    # The whole image moved 150 px left; the object did not move on the ground.
    assert tr.update([(-50, 100)], 0.4, shift=(-150, 0)) == a


def test_two_boxes_keep_their_own_ids():
    tr = CentroidTracker(max_px=60)
    a, b = tr.update([(100, 100), (200, 100)], 0.0)
    assert tr.update([(205, 102), (98, 101)], 0.4) == [b, a]


def test_old_tracks_are_dropped():
    tr = CentroidTracker(max_px=60, max_age_s=1.0)
    a = tr.update([(100, 100)], 0.0)
    assert tr.update([(100, 100)], 2.0) != a


def test_camera_shift_measures_an_image_translation():
    import numpy as np
    from security_drone.detection.confirm import camera_shift
    rng = np.random.default_rng(0)
    big = (rng.random((600, 800, 3)) * 255).astype(np.uint8)
    import cv2
    big = cv2.GaussianBlur(big, (0, 0), 3)
    prev = big[50:530, 50:690]
    frame = big[50:530, 90:730]          # camera moved right: content shifts 40 px left
    dx, dy = camera_shift(prev, frame)
    assert abs(dx + 40) < 3 and abs(dy) < 3


def _frame():
    return np.zeros((96, 128, 3), np.uint8)


def test_filter_hides_a_flicker_and_passes_a_steady_box():
    f = TrackFilter(min_hits=3, window_s=1.0, hold_s=0.5)
    box = Detection(10, 10, 30, 30, 0.5)
    assert f.update(_frame(), [box], 0.0) == []
    assert f.update(_frame(), [box], 0.1) == []
    assert f.update(_frame(), [box], 0.2) == [box]


def test_filter_holds_a_confirmed_box_for_a_moment():
    f = TrackFilter(min_hits=2, window_s=1.0, hold_s=0.5)
    box = Detection(10, 10, 30, 30, 0.5)
    f.update(_frame(), [box], 0.0)
    f.update(_frame(), [box], 0.1)
    assert len(f.update(_frame(), [], 0.3)) == 1
    assert f.update(_frame(), [], 0.8) == []


def test_filter_does_not_hold_an_unconfirmed_box():
    f = TrackFilter(min_hits=3, window_s=1.0, hold_s=0.5)
    f.update(_frame(), [Detection(10, 10, 30, 30, 0.5)], 0.0)
    assert f.update(_frame(), [], 0.1) == []
