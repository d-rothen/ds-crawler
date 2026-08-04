from __future__ import annotations

from pathlib import Path

import pytest

from ds_crawler import (
    DatasetWriter,
    ZipDatasetWriter,
    create_dataset_splits,
    extract_datasets,
    get_dataset_contract,
    get_files,
    index_dataset_from_path,
    list_dataset_splits,
    list_metadata_scopes,
    load_dataset_config,
    load_dataset_split,
    split_dataset,
    split_datasets,
    validate_dataset,
)
from ds_crawler.zip_utils import (
    OUTPUT_FILENAME,
    SCOPES_FILENAME,
    read_metadata_json,
    validate_metadata_scope,
    write_metadata_json_batch,
)

from .current_helpers import create_files, sample_config, sample_head


def _write_scoped_crawler_metadata(
    root: Path,
    *,
    scope: str,
    config: dict,
) -> None:
    on_disk_config = {
        key: value
        for key, value in config.items()
        if key != "head"
    }
    on_disk_config["head_file"] = "dataset-head.json"
    on_disk_config["source"] = {"path": "."}
    write_metadata_json_batch(
        root,
        {
            "dataset-head.json": config["head"],
            "ds-crawler.json": on_disk_config,
        },
        metadata_scope=scope,
    )


def test_index_dataset_from_path_reads_and_writes_scoped_metadata(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    create_files(root, ["frames/0001.png", "frames/0002.png"])
    config = sample_config(
        root,
        name="Scoped RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    _write_scoped_crawler_metadata(root, scope="rgb", config=config)

    output = index_dataset_from_path(root, save_index=True, metadata_scope="rgb")

    assert output["head"]["dataset"]["name"] == "Scoped RGB"
    assert (root / ".ds_crawler" / "rgb" / OUTPUT_FILENAME).is_file()
    assert not (root / ".ds_crawler" / OUTPUT_FILENAME).exists()
    assert read_metadata_json(root, OUTPUT_FILENAME, metadata_scope="rgb") is not None
    assert list_metadata_scopes(root) == ["rgb"]

    manifest = read_metadata_json(root, SCOPES_FILENAME)
    assert manifest["scopes"]["rgb"]["files"] == [
        "dataset-head.json",
        "ds-crawler.json",
        "index.json",
    ]


def test_two_scopes_share_one_physical_root(tmp_path: Path) -> None:
    root = tmp_path / "muses"
    create_files(root, ["frames/0001.png", "calib.json"])
    rgb_config = sample_config(
        root,
        name="MUSES RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    extrinsics_config = sample_config(
        root,
        name="MUSES Camera Extrinsics",
        modality="camera_extrinsics",
        extensions=[".json"],
        id_regex=r"^(calib)\.json$",
    )
    _write_scoped_crawler_metadata(root, scope="rgb", config=rgb_config)
    _write_scoped_crawler_metadata(
        root,
        scope="camera_extrinsics",
        config=extrinsics_config,
    )

    rgb = index_dataset_from_path(root, save_index=True, metadata_scope="rgb")
    extrinsics = index_dataset_from_path(
        root,
        save_index=True,
        metadata_scope="camera_extrinsics",
    )

    assert [entry["path"] for entry in rgb["index"]["files"]] == ["frames/0001.png"]
    assert [entry["path"] for entry in extrinsics["index"]["files"]] == ["calib.json"]
    assert get_dataset_contract(root, metadata_scope="camera_extrinsics").type == "camera_extrinsics"
    assert load_dataset_config(
        {"path": str(root)},
        metadata_scope="rgb",
    ).metadata_scope == "rgb"


def test_scoped_inline_splits_are_isolated(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    create_files(root, ["frames/0001.png", "frames/0002.png"])
    config = sample_config(
        root,
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    _write_scoped_crawler_metadata(root, scope="rgb", config=config)

    index_dataset_from_path(root, save_index=True, metadata_scope="rgb")
    create_dataset_splits(
        root,
        ["train", "val"],
        [0.5, 0.5],
        metadata_scope="rgb",
    )

    assert list_dataset_splits(root) == []
    assert list_dataset_splits(root, metadata_scope="rgb") == ["train", "val"]
    train = load_dataset_split(root, "train", metadata_scope="rgb")
    assert train["split"]["name"] == "train"
    assert read_metadata_json(root, "split_train.json") is None
    assert read_metadata_json(
        root,
        "split_train.json",
        metadata_scope="rgb",
    ) is not None


def test_dataset_writer_saves_scoped_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    writer = DatasetWriter(
        root,
        head=sample_head(name="Predictions", modality="rgb"),
        metadata_scope="pred",
    )
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    assert index_dataset_from_path(root, metadata_scope="pred")["head"]["dataset"]["name"] == "Predictions"
    assert validate_dataset(root, metadata_scope="pred")["metadata_scope"] == "pred"
    assert (root / ".ds_crawler" / "pred" / OUTPUT_FILENAME).is_file()


def test_zip_dataset_writer_saves_scoped_artifacts(tmp_path: Path) -> None:
    zip_path = tmp_path / "predictions.zip"
    with ZipDatasetWriter(
        zip_path,
        head=sample_head(name="Predictions", modality="rgb"),
        metadata_scope="pred",
    ) as writer:
        with writer.open("/scene/0001", "0001.png") as f:
            f.write(b"data")
        writer.save_index()

    loaded = index_dataset_from_path(zip_path, metadata_scope="pred")

    assert loaded["head"]["dataset"]["name"] == "Predictions"
    assert read_metadata_json(zip_path, OUTPUT_FILENAME, metadata_scope="pred") is not None
    assert list_metadata_scopes(zip_path) == ["pred"]


def test_split_dataset_preserves_scoped_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_files(
        source,
        [
            "frames/0001.png",
            "frames/0002.png",
            "frames/0003.png",
            "frames/0004.png",
        ],
    )
    config = sample_config(
        source,
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    _write_scoped_crawler_metadata(source, scope="rgb", config=config)
    index_dataset_from_path(source, save_index=True, metadata_scope="rgb")

    train = tmp_path / "train"
    val = tmp_path / "val"
    split_dataset(source, [0.5, 0.5], [train, val], seed=1, metadata_scope="rgb")

    assert read_metadata_json(train, OUTPUT_FILENAME) is None
    assert read_metadata_json(train, OUTPUT_FILENAME, metadata_scope="rgb") is not None
    train_index = index_dataset_from_path(train, metadata_scope="rgb")
    val_index = index_dataset_from_path(val, metadata_scope="rgb")
    assert len(get_files(train_index)) == 2
    assert len(get_files(val_index)) == 2


def test_split_datasets_preserves_per_source_scopes(tmp_path: Path) -> None:
    root_a = tmp_path / "rgb"
    root_b = tmp_path / "depth"
    create_files(root_a, ["frames/0001.png", "frames/0002.png"])
    create_files(root_b, ["frames/0001.npy", "frames/0002.npy"])
    rgb_config = sample_config(
        root_a,
        name="RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    depth_config = sample_config(
        root_b,
        name="Depth",
        modality="depth",
        extensions=[".npy"],
        id_regex=r"^frames/(.+)\.npy$",
    )
    _write_scoped_crawler_metadata(root_a, scope="rgb", config=rgb_config)
    _write_scoped_crawler_metadata(root_b, scope="depth", config=depth_config)
    index_dataset_from_path(root_a, save_index=True, metadata_scope="rgb")
    index_dataset_from_path(root_b, save_index=True, metadata_scope="depth")

    result = split_datasets(
        [root_a, root_b],
        ["train", "val"],
        [0.5, 0.5],
        seed=1,
        source_metadata_scopes=["rgb", "depth"],
    )

    assert result["per_source"][0]["target_metadata_scope"] == "rgb"
    assert result["per_source"][1]["target_metadata_scope"] == "depth"
    assert read_metadata_json(
        tmp_path / "rgb_train",
        OUTPUT_FILENAME,
        metadata_scope="rgb",
    ) is not None
    assert read_metadata_json(
        tmp_path / "depth_train",
        OUTPUT_FILENAME,
        metadata_scope="depth",
    ) is not None


def test_extract_datasets_writes_scoped_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_files(
        source,
        [
            "frames/0001.png",
            "frames/0001.npy",
            "frames/0002.png",
            "frames/0002.npy",
        ],
    )
    rgb_config = sample_config(
        source,
        name="RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^frames/(.+)\.png$",
    )
    depth_config = sample_config(
        source,
        name="Depth",
        modality="depth",
        extensions=[".npy"],
        id_regex=r"^frames/(.+)\.npy$",
    )
    rgb_target = tmp_path / "rgb"
    depth_target = tmp_path / "depth"

    extract_datasets(
        [rgb_config, depth_config],
        [rgb_target, depth_target],
        metadata_scopes=["rgb", "depth"],
    )

    assert read_metadata_json(rgb_target, OUTPUT_FILENAME) is None
    assert set(get_files(index_dataset_from_path(rgb_target, metadata_scope="rgb"))) == {
        "frames/0001.png",
        "frames/0002.png",
    }
    assert set(get_files(index_dataset_from_path(depth_target, metadata_scope="depth"))) == {
        "frames/0001.npy",
        "frames/0002.npy",
    }


def test_invalid_metadata_scope_raises() -> None:
    with pytest.raises(ValueError, match="metadata_scope"):
        validate_metadata_scope("../rgb")
