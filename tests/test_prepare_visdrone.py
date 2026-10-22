import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "training" / "prepare_visdrone.py"
_spec = importlib.util.spec_from_file_location("prepare_visdrone", _SCRIPT)
prepare_visdrone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_visdrone)


def test_pedestrian_and_people_become_person():
    text = "0 0.1 0.2 0.03 0.05\n1 0.5 0.5 0.02 0.02\n"
    assert prepare_visdrone.to_person_labels(text) == (
        "0 0.1 0.2 0.03 0.05\n0 0.5 0.5 0.02 0.02\n"
    )


def test_vehicles_are_dropped():
    text = "3 0.4 0.4 0.1 0.1\n0 0.1 0.1 0.01 0.03\n9 0.7 0.7 0.02 0.02\n"
    assert prepare_visdrone.to_person_labels(text) == "0 0.1 0.1 0.01 0.03\n"


def test_image_without_person_gets_empty_label():
    assert prepare_visdrone.to_person_labels("3 0.4 0.4 0.1 0.1\n") == ""
    assert prepare_visdrone.to_person_labels("") == ""
