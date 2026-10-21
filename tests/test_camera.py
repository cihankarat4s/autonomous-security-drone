from types import SimpleNamespace

import numpy as np
import pytest

from security_drone.camera.gazebo_camera import BGR_INT8, RGB_INT8, GazeboCamera, image_to_bgr


def fake_image(pixels, fmt=RGB_INT8, padding=0):
    """Build an Image-like message; padding adds unused bytes at the end of each row."""
    height, width, _ = pixels.shape
    rows = np.zeros((height, width * 3 + padding), dtype=np.uint8)
    rows[:, : width * 3] = pixels.reshape(height, width * 3)
    return SimpleNamespace(width=width, height=height, step=rows.shape[1],
                           data=rows.tobytes(), pixel_format_type=fmt)


def test_rgb_is_converted_to_bgr():
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (10, 20, 30)  # R, G, B
    frame = image_to_bgr(fake_image(pixels))
    assert frame.shape == (2, 3, 3)
    assert tuple(frame[0, 0]) == (30, 20, 10)


def test_bgr_is_kept_as_is():
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame = image_to_bgr(fake_image(pixels, fmt=BGR_INT8))
    assert np.array_equal(frame, pixels)


def test_row_padding_is_dropped():
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame = image_to_bgr(fake_image(pixels, fmt=BGR_INT8, padding=5))
    assert np.array_equal(frame, pixels)


def test_unsupported_format_raises():
    msg = fake_image(np.zeros((1, 1, 3), dtype=np.uint8), fmt=4)
    with pytest.raises(ValueError):
        image_to_bgr(msg)


def test_read_returns_latest_frame_or_none():
    camera = GazeboCamera("/topic")
    assert camera.read() is None
    camera._on_image(fake_image(np.ones((2, 2, 3), dtype=np.uint8)))
    assert camera.read().shape == (2, 2, 3)
