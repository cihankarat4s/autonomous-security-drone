import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "sim" / "make_ortho_ground.py"
_spec = importlib.util.spec_from_file_location("make_ortho_ground", _SCRIPT)
make_ortho_ground = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_ortho_ground)

PX = make_ortho_ground.M_PER_PX


def test_home_is_the_world_origin():
    col, row = make_ortho_ground.world_to_source(0, 0)
    assert col * PX == pytest.approx(make_ortho_ground.HOME[0])
    assert row * PX == pytest.approx(make_ortho_ground.HOME[1])


def test_protected_area_is_60_m_north_of_home():
    col, row = make_ortho_ground.world_to_source(0, 60)
    assert col * PX == pytest.approx(make_ortho_ground.AREA_CENTRE[0])
    assert row * PX == pytest.approx(make_ortho_ground.AREA_CENTRE[1])


def test_output_matrix_matches_the_point_mapping():
    m = make_ortho_ground.output_to_source_matrix()
    # Output pixel of world (0, 60): x = 0 is X_MIN + col * PX, y = Y_MAX - row * PX.
    col = -make_ortho_ground.X_MIN / PX
    row = (make_ortho_ground.Y_MAX - 60) / PX
    src = m @ [col, row, 1]
    assert src == pytest.approx(make_ortho_ground.world_to_source(0, 60))
