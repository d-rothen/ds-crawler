from __future__ import annotations

from pathlib import Path

from ds_crawler import (
    build_crawler_config,
    build_dataset_artifacts_from_files,
    build_dataset_head,
)
from ds_crawler.config import DatasetConfig

from .current_helpers import create_files


def test_build_dataset_head_slugifies_id_and_applies_meta_defaults() -> None:
    head = build_dataset_head(
        dataset={
            "name": "Demo RGB",
            "attributes": {"gt": True},
        },
        modality={
            "key": "rgb",
        },
        addons={
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": "demo.input.rgb",
            }
        },
    )

    assert head["dataset"]["id"] == "demo_rgb"
    assert head["dataset"]["attributes"] == {"gt": True}
    assert head["modality"]["meta"]["range"] == [0, 255]
    assert head["addons"]["euler_train"]["slot"] == "demo.input.rgb"


def test_build_dataset_artifacts_from_files_returns_canonical_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    create_files(root, ["Scene01/0001.png", "Scene01/0002.png"])

    bundle = build_dataset_artifacts_from_files(
        dataset={
            "name": "Demo RGB",
            "attributes": {"gt": False},
        },
        modality={
            "key": "rgb",
        },
        addons={
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": "demo.input.rgb",
            }
        },
        indexing={
            "id": {
                "regex": r"^[^/]+/(.+)\.png$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": r"^(?P<scene>[^/]+)/.+\.png$",
                "separator": ":",
            },
            "constraints": {
                "flat_ids_unique": True,
            },
        },
        files=[root / "Scene01/0001.png", root / "Scene01/0002.png"],
        base_path=root,
    )

    artifacts = bundle["artifacts"]
    head = artifacts["dataset-head.json"]
    config = artifacts["ds-crawler.json"]
    index = artifacts["index.json"]

    assert "hydrated_output" not in bundle
    assert bundle["summary"]["dataset_id"] == "demo_rgb"
    assert bundle["summary"]["modality_key"] == "rgb"
    assert bundle["summary"]["file_count"] == 2

    assert head["modality"]["meta"]["file_types"] == ["png"]
    assert config["source"]["path"] == "."
    assert "head" not in config
    assert "head" not in index
    assert "indexing" not in index
    assert index["index"]["children"]["scene:Scene01"]["files"][0]["path"] == (
        "Scene01/0001.png"
    )

    runtime_config = dict(config)
    runtime_config["head"] = head
    dataset_config = DatasetConfig.from_dict(runtime_config, dataset_root=root)
    assert dataset_config.name == "Demo RGB"


def test_build_crawler_config_returns_canonical_mapping() -> None:
    head = build_dataset_head(
        dataset={"name": "Demo Depth"},
        modality={"key": "depth"},
    )

    config = build_crawler_config(
        head=head,
        indexing={
            "id": {
                "regex": r"^(.+)\.npy$",
                "join_char": "+",
            },
            "files": {
                "extensions": [".npy"],
            },
            "constraints": {
                "flat_ids_unique": True,
            },
        },
    )

    assert config["contract"]["kind"] == "ds_crawler_config"
    assert config["head_file"] == "dataset-head.json"
    assert config["source"]["path"] == "."
    assert config["indexing"]["files"]["extensions"] == [".npy"]
