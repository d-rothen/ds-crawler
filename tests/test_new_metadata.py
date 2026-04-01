from __future__ import annotations

import json
import logging
import zipfile

import pytest

from ds_crawler import (
    DatasetWriter,
    create_dataset_splits,
    get_dataset_contract,
    index_dataset_from_files,
    load_dataset_split,
)
from ds_crawler.config import load_dataset_config
from ds_crawler.migration import (
    migrate_dataset_metadata,
    migrate_dataset_zip,
    migrate_dataset_zips_in_folder,
    migrate_inline_splits,
)
from ds_crawler.zip_utils import DATASET_HEAD_FILENAME, OUTPUT_FILENAME, read_metadata_json


def _sample_head() -> dict:
    return {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "demo_rgb", "name": "Demo RGB"},
        "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
        "addons": {
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": "demo.input.rgb",
            }
        },
    }


def _write_legacy_dataset_tree(
    root,
    *,
    path_value: str | None = None,
    include_split: bool = False,
) -> None:
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True)
    (root / "0001.png").write_bytes(b"data")

    legacy_config = {
        "name": "Legacy RGB",
        "path": path_value or str(root),
        "type": "rgb",
        "id_regex": "^(.+)$",
        "properties": {
            "meta": {"range": [0, 255]},
            "euler_train": {
                "used_as": "input",
                "slot": "demo.input.rgb",
                "modality_type": "rgb",
            },
            "gt": False,
        },
    }
    legacy_output = {
        "dataset_contract_version": "1.0",
        "name": "Legacy RGB",
        "type": "rgb",
        "id_regex": "^(.+)$",
        "meta": {"range": [0, 255], "file_types": ["png"]},
        "euler_train": {
            "used_as": "input",
            "slot": "demo.input.rgb",
            "modality_type": "rgb",
        },
        "gt": False,
        "dataset": {"files": [{"path": "0001.png", "id": "0001.png"}]},
    }

    with open(metadata_dir / "ds-crawler.json", "w") as f:
        json.dump(legacy_config, f)
    with open(metadata_dir / "output.json", "w") as f:
        json.dump(legacy_output, f)
    if include_split:
        with open(metadata_dir / "split_train.json", "w") as f:
            json.dump(
                {"files": [{"path": "0001.png", "id": "0001.png"}]},
                f,
            )


def _zip_tree(root, zip_path, *, root_prefix: str = ""):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, root_prefix + str(file.relative_to(root)))
    return zip_path


def _write_legacy_output_only_tree(root) -> None:
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True)
    (root / "0001.png").write_bytes(b"data")

    legacy_output = {
        "dataset_contract_version": "1.0",
        "name": "Foggy RGB",
        "type": "rgb",
        "meta": {"range": [0, 255], "file_types": ["png"]},
        "euler_train": {
            "used_as": "input",
            "slot": "demo.input.rgb",
            "modality_type": "rgb",
        },
        "dataset": {"files": [{"path": "0001.png", "id": "0001.png"}]},
    }
    with open(metadata_dir / "output.json", "w") as f:
        json.dump(legacy_output, f)


def _write_legacy_split(root, name: str = "train") -> None:
    metadata_dir = root / ".ds_crawler"
    with open(metadata_dir / f"split_{name}.json", "w") as f:
        json.dump({"files": [{"path": "0001.png", "id": "0001.png"}]}, f)


def test_index_dataset_from_files_emits_head_and_index(tmp_path) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head": _sample_head(),
        "source": {"path": str(root)},
        "indexing": {
            "id": {"regex": "^(.+)$", "join_char": "+"},
            "constraints": {"flat_ids_unique": True},
        },
    }

    output = index_dataset_from_files(
        config,
        [root / "0001.png", root / "0002.png"],
        base_path=root,
    )

    assert output["contract"]["kind"] == "dataset_index"
    assert output["head"]["dataset"]["id"] == "demo_rgb"
    assert output["index"]["files"][0]["path"] == "0001.png"
    assert output["head"]["modality"]["meta"]["file_types"] == ["png"]


def test_dataset_writer_writes_separate_dataset_head(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "predictions",
        head=_sample_head(),
    )
    path = writer.get_path("/scene:01/0001", "0001.png")
    path.write_bytes(b"data")
    writer.save_index()

    with open(tmp_path / "predictions" / ".ds_crawler" / "dataset-head.json") as f:
        head = json.load(f)
    with open(tmp_path / "predictions" / ".ds_crawler" / "ds-crawler.json") as f:
        config = json.load(f)
    with open(tmp_path / "predictions" / ".ds_crawler" / OUTPUT_FILENAME) as f:
        index = json.load(f)

    assert head["dataset"]["id"] == "demo_rgb"
    assert config["source"]["prebuilt_index_file"] == OUTPUT_FILENAME
    assert "head" not in index
    assert index["index"]["children"]["scene:01"]["files"][0]["path"] == "01/0001.png"


def test_inline_split_files_are_versioned_artifacts(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "predictions",
        head=_sample_head(),
    )
    writer.get_path("/scene:01/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene:01/0002", "0002.png").write_bytes(b"data")
    writer.get_path("/scene:01/0003", "0003.png").write_bytes(b"data")
    writer.save_index()

    result = create_dataset_splits(
        tmp_path / "predictions",
        ["train", "val"],
        [50, 50],
        seed=7,
        sample=2,
    )

    with open(tmp_path / "predictions" / ".ds_crawler" / "split_train.json") as f:
        split_artifact = json.load(f)

    assert result["splits"][0]["metadata_file"] == ".ds_crawler/split_train.json"
    assert split_artifact["contract"]["kind"] == "dataset_split"
    assert split_artifact["split"]["name"] == "train"
    assert split_artifact["split"]["source_index_file"] == OUTPUT_FILENAME
    assert split_artifact["execution"] == {
        "ratio": 50,
        "seed": 7,
        "sampled": 2,
    }
    assert "head" not in split_artifact
    assert "index" in split_artifact

    loaded = load_dataset_split(tmp_path / "predictions", "train")
    assert loaded["head"]["dataset"]["id"] == "demo_rgb"
    assert loaded["split"]["name"] == "train"
    assert loaded["execution"]["split"] == split_artifact["execution"]
    assert loaded["index"] == split_artifact["index"]


def test_migrate_metadata_rewrites_legacy_files(tmp_path) -> None:
    legacy_config = {
        "name": "Legacy RGB",
        "path": str(tmp_path),
        "type": "rgb",
        "id_regex": "^(.+)$",
        "properties": {
            "meta": {"range": [0, 255]},
            "euler_train": {
                "used_as": "input",
                "slot": "demo.input.rgb",
                "modality_type": "rgb",
            },
            "gt": False,
        },
    }
    legacy_output = {
        "dataset_contract_version": "1.0",
        "name": "Legacy RGB",
        "type": "rgb",
        "id_regex": "^(.+)$",
        "meta": {"range": [0, 255], "file_types": ["png"]},
        "euler_train": {
            "used_as": "input",
            "slot": "demo.input.rgb",
            "modality_type": "rgb",
        },
        "gt": False,
        "dataset": {"files": [{"path": "0001.png", "id": "0001.png"}]},
    }
    (tmp_path / "0001.png").write_bytes(b"data")
    with open(tmp_path / "ds-crawler.json", "w") as f:
        json.dump(legacy_config, f)
    with open(tmp_path / "output.json", "w") as f:
        json.dump(legacy_output, f)

    result = migrate_dataset_metadata(tmp_path)
    contract = get_dataset_contract(tmp_path)

    assert result["wrote_head"] is True
    assert contract.dataset_id == "legacy_rgb"
    assert contract.type == "rgb"


def test_migrate_dataset_zip_rewrites_prefixed_archive(tmp_path) -> None:
    root = tmp_path / "legacy_rgb_tree"
    root.mkdir()
    _write_legacy_dataset_tree(root, path_value=".", include_split=True)

    zip_path = _zip_tree(
        root,
        tmp_path / "legacy_rgb.zip",
        root_prefix="legacy_rgb/",
    )

    result = migrate_dataset_zip(
        zip_path,
        logger=logging.getLogger("tests.test_new_metadata"),
    )

    head = read_metadata_json(zip_path, DATASET_HEAD_FILENAME)
    output = read_metadata_json(zip_path, OUTPUT_FILENAME)
    split = read_metadata_json(zip_path, "split_train.json")

    assert result["wrote_head"] is True
    assert head is not None
    assert head["dataset"]["id"] == "legacy_rgb"
    assert output is not None
    assert output["contract"]["kind"] == "dataset_index"
    assert "head" not in output
    assert output["index"]["files"][0]["path"] == "0001.png"
    assert split is not None
    assert split["contract"]["kind"] == "dataset_split"
    assert split["split"]["name"] == "train"
    assert split["split"]["source_index_file"] == OUTPUT_FILENAME
    assert split["index"]["files"][0]["path"] == "0001.png"


def test_migrate_dataset_zips_in_folder_logs_missing_metadata(tmp_path, caplog) -> None:
    valid_root = tmp_path / "nested" / "valid_tree"
    valid_root.parent.mkdir(parents=True)
    valid_root.mkdir()
    _write_legacy_dataset_tree(valid_root, path_value=".")
    _zip_tree(
        valid_root,
        tmp_path / "nested" / "valid_rgb.zip",
        root_prefix="valid_rgb/",
    )

    invalid_root = tmp_path / "other" / "invalid_tree"
    invalid_root.parent.mkdir(parents=True)
    invalid_root.mkdir()
    (invalid_root / "0001.png").write_bytes(b"data")
    _zip_tree(
        invalid_root,
        tmp_path / "other" / "invalid_rgb.zip",
        root_prefix="invalid_rgb/",
    )

    logger = logging.getLogger("ds_crawler.migration")
    with caplog.at_level(logging.INFO, logger="ds_crawler.migration"):
        result = migrate_dataset_zips_in_folder(
            tmp_path,
            logger=logger,
        )

    assert result["scanned"] == 2
    assert len(result["migrated"]) == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"].endswith("invalid_rgb.zip")
    assert result["recursive"] is True
    assert ".ds_crawler/" in caplog.text


def test_migrate_output_only_zip_without_id_regex(tmp_path) -> None:
    root = tmp_path / "foggy_rgb_tree"
    root.mkdir()
    _write_legacy_output_only_tree(root)

    zip_path = _zip_tree(
        root,
        tmp_path / "foggy_rgb.zip",
        root_prefix="foggy_rgb/",
    )

    result = migrate_dataset_zip(zip_path)
    config = read_metadata_json(zip_path, "ds-crawler.json")
    assert result["wrote_config"] is True
    assert config is not None
    assert config["source"]["prebuilt_index_file"] == OUTPUT_FILENAME
    assert config["indexing"].get("id") is None

    loaded_config = load_dataset_config({"path": str(zip_path)})
    assert loaded_config.id_regex is None
    assert loaded_config.prebuilt_index_file is not None


def test_migrate_split_without_index_fails_for_prebuilt_dataset(tmp_path) -> None:
    root = tmp_path / "foggy_rgb_tree"
    root.mkdir()
    _write_legacy_output_only_tree(root)
    _write_legacy_split(root)

    with pytest.raises(
        ValueError,
        match="Cannot migrate split metadata without writing index.json",
    ):
        migrate_dataset_metadata(root, write_output=False)


def test_migrate_inline_splits_converts_stale_splits(tmp_path) -> None:
    """Partial migration: index.json is new-schema but split files are legacy."""
    root = tmp_path / "partial"
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True)

    # Write already-migrated index.json
    new_index = {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "index": {"files": [{"path": "0001.png", "id": "0001.png"}]},
        "generator": {"name": "ds_crawler", "version": "2.0"},
        "execution": {},
    }
    with open(metadata_dir / "index.json", "w") as f:
        json.dump(new_index, f)

    # Write stale (legacy) split files
    _write_legacy_split(root, "train")
    _write_legacy_split(root, "val")

    result = migrate_inline_splits(root)

    assert len(result["migrated_splits"]) == 2
    assert "split_train.json" in result["migrated_splits"]
    assert "split_val.json" in result["migrated_splits"]

    with open(metadata_dir / "split_train.json") as f:
        split_artifact = json.load(f)

    assert split_artifact["contract"]["kind"] == "dataset_split"
    assert split_artifact["split"]["name"] == "train"
    assert split_artifact["split"]["source_index_file"] == OUTPUT_FILENAME
    assert split_artifact["index"]["files"][0]["path"] == "0001.png"


def test_migrate_inline_splits_skips_already_migrated(tmp_path) -> None:
    """Already-migrated splits are counted but not rewritten."""
    root = tmp_path / "partial"
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True)

    new_index = {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "index": {"files": [{"path": "0001.png", "id": "0001.png"}]},
        "generator": {"name": "ds_crawler", "version": "2.0"},
        "execution": {},
    }
    with open(metadata_dir / "index.json", "w") as f:
        json.dump(new_index, f)

    new_split = {
        "contract": {"kind": "dataset_split", "version": "1.0"},
        "split": {"name": "train", "source_index_file": "index.json"},
        "index": {"files": [{"path": "0001.png", "id": "0001.png"}]},
    }
    with open(metadata_dir / "split_train.json", "w") as f:
        json.dump(new_split, f)

    result = migrate_inline_splits(root)

    assert result["migrated_splits"] == ["split_train.json"]
    # File should remain unchanged
    with open(metadata_dir / "split_train.json") as f:
        assert json.load(f) == new_split


def test_migrate_inline_splits_fails_without_index(tmp_path) -> None:
    """Cannot run inline-splits migration if index.json is missing."""
    root = tmp_path / "no_index"
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True)

    _write_legacy_split(root, "train")

    with pytest.raises(FileNotFoundError, match="index.json not found"):
        migrate_inline_splits(root)
