from __future__ import annotations

import json
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


def test_build_dataset_artifacts_from_files_returns_scoped_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    create_files(root, ["Scene01/0001.png"])

    bundle = build_dataset_artifacts_from_files(
        dataset={"name": "Demo RGB"},
        modality={"key": "rgb"},
        indexing={
            "id": {
                "regex": r"^[^/]+/(.+)\.png$",
                "join_char": "+",
            },
            "constraints": {
                "flat_ids_unique": True,
            },
        },
        files=[root / "Scene01/0001.png"],
        base_path=root,
        metadata_scope="rgb",
    )

    artifacts = bundle["artifacts"]

    assert bundle["summary"]["metadata_scope"] == "rgb"
    assert set(artifacts) == {
        "rgb/dataset-head.json",
        "rgb/ds-crawler.json",
        "rgb/index.json",
        "scopes.json",
    }
    assert artifacts["rgb/ds-crawler.json"]["metadata_scope"] == "rgb"
    assert artifacts["scopes.json"]["scopes"]["rgb"]["metadata_dir"] == (
        ".ds_crawler/rgb"
    )
    assert artifacts["scopes.json"]["scopes"]["rgb"]["files"] == [
        "dataset-head.json",
        "ds-crawler.json",
        "index.json",
    ]


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


def _entry(path: str, file_id: str, **attributes) -> dict:
    entry = {"path": path, "id": file_id, "path_properties": {}, "basename_properties": {}}
    if attributes:
        entry["attributes"] = attributes
    return entry


def test_save_dataset_index_writes_a_prebuilt_scope(tmp_path: Path) -> None:
    from ds_crawler import (
        build_index_output,
        get_files,
        index_dataset_from_path,
        list_metadata_scopes,
        save_dataset_index,
        validate_dataset,
    )

    create_files(tmp_path, ["sweeps/lidar_100.bin", "sweeps/lidar_150.bin"])
    head = build_dataset_head(
        dataset={"id": "demo_lidar", "name": "Demo lidar"},
        modality={"key": "lidar_point_cloud"},
    )
    # IDs that no path regex could produce: the camera frame each sweep pairs with.
    index = {
        "children": {
            "scene:a": {
                "files": [
                    _entry("sweeps/lidar_100.bin", "57", offset_ms=43.0),
                    _entry("sweeps/lidar_150.bin", "107", offset_ms=43.0),
                ]
            }
        }
    }
    output = build_index_output(
        head=head,
        index=index,
        generator={"name": "demo-generator", "version": "1.0"},
        execution={"recipe": "demo"},
    )
    path = save_dataset_index(tmp_path, output, metadata_scope="lidar")

    assert path == tmp_path / ".ds_crawler" / "lidar" / "index.json"
    config = json.loads((tmp_path / ".ds_crawler" / "lidar" / "ds-crawler.json").read_text())
    assert config["source"]["prebuilt_index_file"] == "index.json"
    artifact = json.loads(path.read_text())
    assert artifact["generator"] == {"name": "demo-generator", "version": "1.0"}
    assert artifact["execution"] == {"recipe": "demo"}
    assert "lidar" in list_metadata_scopes(tmp_path)
    validate_dataset(tmp_path, metadata_scope="lidar")

    loaded = index_dataset_from_path(tmp_path, metadata_scope="lidar")
    files = loaded["index"]["children"]["scene:a"]["files"]
    assert [entry["id"] for entry in files] == ["57", "107"]
    assert files[0]["attributes"] == {"offset_ms": 43.0}
    assert get_files(loaded) == ["sweeps/lidar_100.bin", "sweeps/lidar_150.bin"]


def test_prebuilt_scope_keeps_its_hierarchy_separator(tmp_path: Path) -> None:
    """A prebuilt index may declare the separator of its name:value keys
    without a regex; loading it back keeps the separator."""
    from ds_crawler import (
        build_index_output,
        index_dataset_from_path,
        save_dataset_index,
    )

    create_files(tmp_path, ["a/0.png"])
    head = build_dataset_head(dataset={"id": "demo", "name": "demo"}, modality={"key": "rgb"})
    output = build_index_output(
        head=head,
        index={"children": {"scene:a": {"files": [_entry("a/0.png", "0")]}}},
        indexing={"hierarchy": {"separator": ":"}},
    )
    save_dataset_index(tmp_path, output, metadata_scope="rgb")
    config = json.loads((tmp_path / ".ds_crawler" / "rgb" / "ds-crawler.json").read_text())
    assert config["indexing"] == {"hierarchy": {"separator": ":"}}
    assert config["source"]["prebuilt_index_file"] == "index.json"
    loaded = index_dataset_from_path(tmp_path, metadata_scope="rgb")
    assert loaded["indexing"]["hierarchy"] == {"separator": ":"}


def test_save_dataset_index_keeps_regex_indexing(tmp_path: Path) -> None:
    from ds_crawler import (
        index_dataset_from_files,
        index_dataset_from_path,
        save_dataset_index,
    )

    source = tmp_path / "source"
    create_files(source, ["scene_01/0001.png", "scene_01/0002.png"])
    head = build_dataset_head(dataset={"id": "demo_rgb", "name": "Demo RGB"}, modality={"key": "rgb"})
    config = build_crawler_config(
        head=head,
        indexing={
            "id": {"regex": r"^[^/]+/(?P<frame>\d+)\.png$", "join_char": "+"},
            "hierarchy": {"regex": r"^(?P<scene>[^/]+)/", "separator": ":"},
        },
    )
    config["head"] = head
    output = index_dataset_from_files(
        config, sorted(source.rglob("*.png")), base_path=source
    )

    # The index can live in another root than the one that was crawled.
    target = tmp_path / "target"
    target.mkdir()
    save_dataset_index(target, output)
    saved = json.loads((target / ".ds_crawler" / "ds-crawler.json").read_text())
    assert "prebuilt_index_file" not in saved["source"]
    assert saved["indexing"]["id"]["regex"] == r"^[^/]+/(?P<frame>\d+)\.png$"
    loaded = index_dataset_from_path(target)
    assert sorted(loaded["index"]["children"]["scene:scene_01"]["files"][i]["id"] for i in range(2)) == [
        "frame-0001",
        "frame-0002",
    ]


def test_invalid_index_output_writes_nothing(tmp_path: Path) -> None:
    import pytest

    from ds_crawler import build_index_output, save_dataset_index

    head = build_dataset_head(dataset={"id": "demo", "name": "Demo"}, modality={"key": "rgb"})
    with pytest.raises(ValueError):
        build_index_output(head=head, index={"files": [{"path": "a.png"}]})
    with pytest.raises(ValueError, match="generator"):
        build_index_output(head=head, index={}, generator={"name": ""})
    bad = {"contract": {"kind": "dataset_index", "version": "1.0"}, "head": {}, "index": {}}
    with pytest.raises(ValueError):
        save_dataset_index(tmp_path, bad)
    assert not (tmp_path / ".ds_crawler").exists()
