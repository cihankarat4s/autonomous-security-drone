import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_visdrone_multiclass", Path(__file__).parents[1] / "training/prepare_visdrone_multiclass.py")
pvm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvm)


def test_classes_are_merged():
    text = "0 .1 .1 .1 .1\n1 .2 .2 .2 .2\n9 .3 .3 .3 .3\n2 .4 .4 .4 .4\n8 .5 .5 .5 .5\n"
    assert [r.split()[0] for r in pvm.remap_labels(text).splitlines()] == ["0", "0", "1", "2", "3"]
