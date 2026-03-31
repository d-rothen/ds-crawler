from __future__ import annotations

from pathlib import Path

from ds_crawler import extract_datasets, get_files, index_dataset_from_path

from .current_helpers import create_files, sample_config


def test_extract_datasets_writes_split_outputs_for_each_config(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_files(
        source,
        [
            "scene/0001.png",
            "scene/0001.npy",
            "scene/0002.png",
            "scene/0002.npy",
        ],
    )

    rgb_config = sample_config(
        source,
        name="rgb",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
    )
    depth_config = sample_config(
        source,
        name="depth",
        modality="depth",
        extensions=[".npy"],
        id_regex=r"^(.+)\.npy$",
    )

    rgb_target = tmp_path / "rgb"
    depth_target = tmp_path / "depth"
    result = extract_datasets(
        [rgb_config, depth_config],
        [rgb_target, depth_target],
    )

    assert result["common_ids"] == {("scene/0001",), ("scene/0002",)}
    assert set(get_files(index_dataset_from_path(rgb_target))) == {
        "scene/0001.png",
        "scene/0002.png",
    }
    assert set(get_files(index_dataset_from_path(depth_target))) == {
        "scene/0001.npy",
        "scene/0002.npy",
    }


def test_extract_datasets_reports_incomplete_intersection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_files(
        source,
        [
            "scene/0001.png",
            "scene/0002.png",
            "scene/0001.npy",
        ],
    )

    rgb_config = sample_config(
        source,
        name="rgb",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
    )
    depth_config = sample_config(
        source,
        name="depth",
        modality="depth",
        extensions=[".npy"],
        id_regex=r"^(.+)\.npy$",
    )

    result = extract_datasets(
        [rgb_config, depth_config],
        [tmp_path / "rgb", tmp_path / "depth"],
    )

    assert result["common_ids"] == {("scene/0001",)}
    assert result["incomplete_ids"]["rgb"] == {("scene/0002",)}
