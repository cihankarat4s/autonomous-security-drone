import numpy as np

from security_drone.detection.preprocess import preprocess


def test_preprocess_keeps_shape_and_dtype():
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    out = preprocess(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_equalize_spreads_dark_image():
    # A dim, low-contrast frame should use more of the 0-255 range afterwards.
    frame = np.tile(np.linspace(40, 80, 64, dtype=np.uint8), (48, 1))
    frame = np.dstack([frame] * 3)
    out = preprocess(frame, equalize=True, sharpen_edges=False)
    assert np.ptp(out) > np.ptp(frame)


def test_disabled_steps_return_input():
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert preprocess(frame, equalize=False, sharpen_edges=False) is frame
