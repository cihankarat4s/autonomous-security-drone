import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_semantic_drone", Path(__file__).parents[1] / "training/prepare_semantic_drone.py")
sds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sds)


def test_blocks_keep_neighbouring_images_together():
    stems = [f"{i:03d}" for i in range(400)]
    splits = sds.split_blocks(stems)
    assert len(splits["test"]) == 100 and len(splits["val"]) == 60
    assert sorted(sum(splits.values(), [])) == stems
    assert splits["test"][:20] == stems[60:80]


def test_tiles_cover_the_image():
    origins = sds.tile_origins(6000, 4000)
    assert len(origins) == 9
    assert origins[-1] == (3000, 2000)


def test_crop_drops_boxes_mostly_outside():
    assert len(sds.crop_labels([(90, 0, 110, 20)], 0, 0, 100, 100)) == 1
    assert sds.crop_labels([(95, 0, 115, 20)], 0, 0, 100, 100) == []
    assert sds.crop_labels([(200, 0, 220, 20)], 0, 0, 100, 100) == []
