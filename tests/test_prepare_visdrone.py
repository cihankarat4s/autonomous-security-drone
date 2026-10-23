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


def test_negatives_get_empty_labels(tmp_path):
    negatives = tmp_path / "neg"
    negatives.mkdir()
    (negatives / "sun_a_0000_alt06.0.jpg").write_bytes(b"jpg")
    target = tmp_path / "ds"
    (target / "images" / "train").mkdir(parents=True)
    (target / "labels" / "train").mkdir(parents=True)

    assert prepare_visdrone.add_negatives(target, [negatives]) == 1
    assert (target / "images" / "train" / "sun_a_0000_alt06.0.jpg").read_bytes() == b"jpg"
    assert (target / "labels" / "train" / "sun_a_0000_alt06.0.txt").read_text() == ""
