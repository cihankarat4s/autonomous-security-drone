import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_extern", Path(__file__).parents[1] / "training/prepare_extern.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)


def test_cyclists_stay_people():
    text = "0 .1 .1 .1 .1\n1 .2 .2 .2 .2\n2 .3 .3 .3 .3\n"
    assert [r.split()[0] for r in pe.remap(text, pe.JLU_MAP).splitlines()] == ["3", "0", "0"]
