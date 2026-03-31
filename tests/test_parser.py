from __future__ import annotations

from pathlib import Path

import pytest
from ds_crawler import get_files, index_dataset_from_files, index_dataset_from_path

from .current_helpers import create_files, sample_config, write_crawler_metadata


def test_index_dataset_from_files_emits_current_output_shape(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene/0001.png", "scene/0002.png"])

    output = index_dataset_from_files(
        sample_config(
            root,
            name="RGB",
            modality="rgb",
            extensions=[".png"],
            id_regex=r"^(.+)\.png$",
        ),
        files,
        base_path=root,
    )

    assert output["contract"]["kind"] == "dataset_index"
    assert output["head"]["dataset"]["name"] == "RGB"
    assert output["head"]["modality"]["key"] == "rgb"
    assert output["head"]["modality"]["meta"]["file_types"] == ["png"]
    assert [entry["path"] for entry in output["index"]["files"]] == [
        "scene/0001.png",
        "scene/0002.png",
    ]


def test_index_dataset_from_path_writes_canonical_metadata_files(tmp_path: Path) -> None:
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

    output = index_dataset_from_path(root, save_index=True)

    assert (root / ".ds_crawler" / "dataset-head.json").is_file()
    assert (root / ".ds_crawler" / "ds-crawler.json").is_file()
    assert (root / ".ds_crawler" / "index.json").is_file()
    assert output["head"]["dataset"]["id"] == "rgb"
    assert output["index"]["files"][0]["id"] == "scene/0001"


def test_index_dataset_from_files_sampling_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(
        root,
        [f"scene/{frame:04d}.png" for frame in range(1, 6)],
    )

    output = index_dataset_from_files(
        sample_config(root, extensions=[".png"], id_regex=r"^(.+)\.png$"),
        files,
        base_path=root,
        sample=2,
    )

    assert output["execution"]["sampled"] == 2
    assert [entry["id"] for entry in output["index"]["files"]] == [
        "scene/0001",
        "scene/0003",
        "scene/0005",
    ]


def test_index_dataset_from_files_match_index_filters_by_qualified_id(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    all_files = create_files(
        root,
        ["scene_a/0001.png", "scene_b/0001.png", "scene_b/0002.png"],
    )
    config = sample_config(
        root,
        extensions=[".png"],
        id_regex=r"^[^/]+/(.+)\.png$",
        hierarchy_regex=r"^(?P<scene>[^/]+)/.+\.png$",
    )
    match_index = index_dataset_from_files(
        config,
        [root / "scene_b/0001.png"],
        base_path=root,
    )

    output = index_dataset_from_files(
        config,
        all_files,
        base_path=root,
        match_index=match_index,
    )

    assert set(output["index"]["children"]) == {"scene:scene_b"}
    assert [entry["path"] for entry in output["index"]["children"]["scene:scene_b"]["files"]] == [
        "scene_b/0001.png"
    ]


def test_index_dataset_from_files_skips_flat_duplicates_by_default(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene_a/0001.png", "scene_b/0001.png"])
    config = sample_config(
        root,
        extensions=[".png"],
        id_regex=r"^[^/]+/(.+)\.png$",
        hierarchy_regex=r"^(?P<scene>[^/]+)/.+\.png$",
    )

    output = index_dataset_from_files(config, files, base_path=root)

    assert get_files(output) == ["scene_a/0001.png"]


def test_index_dataset_from_files_strict_duplicate_raises(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene_a/0001.png", "scene_b/0001.png"])
    config = sample_config(
        root,
        extensions=[".png"],
        id_regex=r"^[^/]+/(.+)\.png$",
        hierarchy_regex=r"^(?P<scene>[^/]+)/.+\.png$",
    )

    with pytest.raises(RuntimeError, match="Duplicate id"):
        index_dataset_from_files(config, files, base_path=root, strict=True)


def test_index_dataset_from_files_extracts_path_and_basename_properties(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene_a/0001.png"])
    config = sample_config(
        root,
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
        basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
        path_regex=r"^(?P<scene>[^/]+)/",
    )

    output = index_dataset_from_files(config, files, base_path=root)
    entry = output["index"]["files"][0]

    assert entry["path_properties"] == {"scene": "scene_a"}
    assert entry["basename_properties"] == {"frame": "0001", "ext": "png"}
