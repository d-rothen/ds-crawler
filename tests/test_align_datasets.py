from __future__ import annotations

from pathlib import Path

from ds_crawler import align_datasets, index_dataset_from_files, index_dataset_from_path

from .current_helpers import create_files, sample_config, write_crawler_metadata


def test_align_datasets_with_output_dicts(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    rgb_files = create_files(rgb_root, ["scene/0001.png", "scene/0002.png"])
    depth_files = create_files(depth_root, ["scene/0001.npy", "scene/0002.npy"])

    rgb_output = index_dataset_from_files(
        sample_config(
            rgb_root,
            name="RGB",
            modality="rgb",
            extensions=[".png"],
            id_regex=r"^(.+)\.png$",
        ),
        rgb_files,
        base_path=rgb_root,
    )
    depth_output = index_dataset_from_files(
        sample_config(
            depth_root,
            name="Depth",
            modality="depth",
            extensions=[".npy"],
            id_regex=r"^(.+)\.npy$",
        ),
        depth_files,
        base_path=depth_root,
    )

    aligned = align_datasets(
        {"modality": "rgb", "source": rgb_output},
        {"modality": "depth", "source": depth_output},
    )

    assert set(aligned) == {"scene/0001", "scene/0002"}
    assert aligned["scene/0001"]["rgb"]["path"] == "scene/0001.png"
    assert aligned["scene/0001"]["depth"]["path"] == "scene/0001.npy"


def test_align_datasets_with_dataset_paths(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    create_files(root, ["scene/0001.png", "scene/0002.png"])
    config = sample_config(
        root,
        name="RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
    )
    write_crawler_metadata(root, config)
    index_dataset_from_path(root, save_index=True)

    aligned = align_datasets({"modality": "rgb", "source": root})

    assert set(aligned) == {"scene/0001", "scene/0002"}
    assert aligned["scene/0002"]["rgb"]["path"] == "scene/0002.png"
