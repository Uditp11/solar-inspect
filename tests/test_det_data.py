"""The val-is-not-test guard, the checkpoint-hash guard and the label-to-pixel conversion,
on files made here."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from solar_inspect.detection.data import check_checkpoint, ground_truth, splits


def write_yaml(tmp_path, val: str):
    p = tmp_path / "configs" / "d2.yaml"
    p.parent.mkdir()
    p.write_text(f"train: ../d/train/images\nval: {val}\ntest: ../d/test/images\n")
    return p


def test_the_committed_layout_passes(tmp_path):
    p = splits(write_yaml(tmp_path, "../d/val/images"))
    assert p["val"] == (tmp_path / "d" / "val" / "images").resolve()


@pytest.mark.parametrize("val", ["../d/test/images", "../d/test/images/../images",
                                 "../d/test", "../d/test/images/sub"])
def test_val_resolving_to_the_test_tree_is_refused(tmp_path, val):
    with pytest.raises(SystemExit):
        splits(write_yaml(tmp_path, val))


def test_ground_truth_converts_normalised_centres_to_pixel_corners(tmp_path):
    # 640 x 512 frame, box centre (0.5, 0.25), size (0.1, 0.2) of the frame:
    # centre (320, 128), size (64, 102.4) -> [288, 76.8, 352, 179.2]
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (640, 512)).save(tmp_path / "images" / "a.jpg")
    Image.new("RGB", (640, 512)).save(tmp_path / "images" / "b.jpg")   # no label: background
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.25 0.1 0.2\n")
    names, boxes = ground_truth(tmp_path / "images")
    assert names == ["a.jpg", "b.jpg"]
    np.testing.assert_allclose(boxes[0], [[288, 76.8, 352, 179.2]])
    assert boxes[1].shape == (0, 4)


# SHA-256 of the three bytes b"abc" (FIPS 180-2, appendix B.1)
ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_a_checkpoint_with_the_pinned_hash_passes(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"abc")
    assert check_checkpoint(tmp_path / "best.pt", ABC) == ABC


@pytest.mark.parametrize("pinned", [ABC[:-1] + "0", "", None])
def test_a_checkpoint_with_another_hash_or_none_pinned_is_refused(tmp_path, pinned):
    (tmp_path / "best.pt").write_bytes(b"abc")
    with pytest.raises(SystemExit):
        check_checkpoint(tmp_path / "best.pt", pinned)


def test_a_changed_checkpoint_is_refused(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"abd")
    with pytest.raises(SystemExit):
        check_checkpoint(tmp_path / "best.pt", ABC)
