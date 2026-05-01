from __future__ import annotations

import pytest

from ds_crawler import DatasetWriter, build_layout_addon, get_layout_addon
from ds_crawler.zip_utils import DATASET_HEAD_FILENAME, read_metadata_json


def test_build_layout_addon_validates_axes() -> None:
    addon = build_layout_addon(
        family="demo",
        sample_axis_name="file_id",
        sample_axis_location="hierarchy",
        variant_axis_name="fog_augmentation",
        variant_axis_location="file_id",
        derived_from={"source_modality": "rgb"},
    )

    assert addon == {
        "version": "1.0",
        "family": "demo",
        "sample_axis": {"name": "file_id", "location": "hierarchy"},
        "variant_axis": {"name": "fog_augmentation", "location": "file_id"},
        "derived_from": {"source_modality": "rgb"},
    }


def test_build_layout_addon_rejects_unknown_axis_location() -> None:
    with pytest.raises(ValueError, match="location"):
        build_layout_addon(
            sample_axis_name="file_id",
            sample_axis_location="directory",
        )


def test_dataset_writer_persists_euler_layout_addon(tmp_path) -> None:
    root = tmp_path / "aug_rgb"
    layout = build_layout_addon(
        sample_axis_name="file_id",
        sample_axis_location="hierarchy",
        variant_axis_name="fog_augmentation",
        variant_axis_location="file_id",
    )
    writer = DatasetWriter(
        root,
        name="Augmented RGB",
        type="rgb",
        meta={"range": [0, 255]},
        euler_layout=layout,
    )
    path = writer.get_path("/scene:Scene01/file_id:0001/mor_10m", "mor_10m.png")
    path.write_bytes(b"")
    writer.save_index()

    head = read_metadata_json(root, DATASET_HEAD_FILENAME)

    assert head is not None
    assert head["addons"]["euler_layout"] == layout
    assert get_layout_addon({"head": head}) == layout
