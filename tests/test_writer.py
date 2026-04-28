from __future__ import annotations

from pathlib import Path

from ds_crawler import DatasetWriter, ZipDatasetWriter, index_dataset_from_path
from ds_crawler.zip_utils import OUTPUT_FILENAME, read_metadata_json

from .current_helpers import sample_head


def test_dataset_writer_saves_canonical_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Segmentation", modality="rgb"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    loaded = index_dataset_from_path(root)

    assert (root / ".ds_crawler" / "dataset-head.json").is_file()
    assert (root / ".ds_crawler" / "ds-crawler.json").is_file()
    assert (root / ".ds_crawler" / OUTPUT_FILENAME).is_file()
    assert loaded["head"]["dataset"]["name"] == "Segmentation"


def test_zip_dataset_writer_saves_canonical_artifacts(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with ZipDatasetWriter(zip_path, head=sample_head(name="Segmentation", modality="rgb")) as writer:
        with writer.open("/scene/0001", "0001.png") as f:
            f.write(b"data")
        writer.save_index()

    index_artifact = read_metadata_json(zip_path, OUTPUT_FILENAME)
    loaded = index_dataset_from_path(zip_path)

    assert index_artifact is not None
    assert "head" not in index_artifact
    assert loaded["head"]["dataset"]["name"] == "Segmentation"


def test_dataset_writer_get_path_uses_hierarchy_values_and_source_meta(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Segmentation", modality="rgb"))
    path = writer.get_path(
        "/scene:Scene01/cam:Cam0/0001",
        "0001.png",
        source_meta={
            "path_properties": {"scene": "override_scene"},
            "basename_properties": {"frame": "0001", "ext": "png"},
        },
    )
    path.write_bytes(b"data")
    output = writer.build_output()

    entry = output["index"]["children"]["scene:Scene01"]["children"]["cam:Cam0"]["files"][0]

    assert path == root / "Scene01" / "Cam0" / "0001.png"
    assert entry["id"] == "0001"
    assert entry["path_properties"] == {"scene": "override_scene"}
    assert entry["basename_properties"] == {"frame": "0001", "ext": "png"}


# ---------------------------------------------------------------------------
# Per-file ``attributes`` field
# ---------------------------------------------------------------------------


def _first_file_entry(output: dict, *child_keys: str) -> dict:
    node = output["index"]
    for key in child_keys:
        node = node["children"][key]
    return node["files"][0]


def test_dataset_writer_get_path_records_attributes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene/0001", "0001.png",
        attributes={"checkpoint": "v3.2", "noise_sigma": 0.1, "tags": ["aug"]},
    ).write_bytes(b"data")
    entry = _first_file_entry(writer.build_output(), "scene")
    assert entry["attributes"] == {
        "checkpoint": "v3.2",
        "noise_sigma": 0.1,
        "tags": ["aug"],
    }


def test_dataset_writer_omits_attributes_when_none_or_empty(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png", attributes={}).write_bytes(b"data")

    files = writer.build_output()["index"]["children"]["scene"]["files"]
    assert "attributes" not in files[0]
    assert "attributes" not in files[1]


def test_dataset_writer_inherits_attributes_from_source_entry(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene/0001", "0001.png",
        source_entry={
            "path_properties": {},
            "basename_properties": {},
            "attributes": {"src": "blender"},
        },
    ).write_bytes(b"data")
    entry = _first_file_entry(writer.build_output(), "scene")
    assert entry["attributes"] == {"src": "blender"}


def test_explicit_attributes_override_source_entry_attributes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene/0001", "0001.png",
        source_entry={
            "path_properties": {},
            "basename_properties": {},
            "attributes": {"src": "blender", "exposure": 0.7},
        },
        attributes={"src": "houdini"},  # full replace, no merge
    ).write_bytes(b"data")
    entry = _first_file_entry(writer.build_output(), "scene")
    assert entry["attributes"] == {"src": "houdini"}


def test_source_meta_alias_still_works(tmp_path: Path) -> None:
    """``source_meta`` is the deprecated kwarg name for ``source_entry``."""
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene/0001", "0001.png",
        source_meta={
            "path_properties": {"scene": "Scene01"},
            "basename_properties": {"ext": "png"},
            "attributes": {"src": "deprecated_kwarg"},
        },
    ).write_bytes(b"data")
    entry = _first_file_entry(writer.build_output(), "scene")
    assert entry["path_properties"] == {"scene": "Scene01"}
    assert entry["attributes"] == {"src": "deprecated_kwarg"}


def test_passing_both_source_entry_and_source_meta_raises(tmp_path: Path) -> None:
    import pytest

    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    with pytest.raises(TypeError):
        writer.get_path(
            "/scene/0001", "0001.png",
            source_entry={"path_properties": {}, "basename_properties": {}},
            source_meta={"path_properties": {}, "basename_properties": {}},
        )


def test_zip_writer_open_records_attributes(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with ZipDatasetWriter(zip_path, head=sample_head(name="Aug", modality="rgb")) as writer:
        with writer.open(
            "/scene/0001", "0001.png", attributes={"k": "v"},
        ) as f:
            f.write(b"data")
        writer.save_index()

    loaded = index_dataset_from_path(zip_path)
    entry = _first_file_entry(loaded, "scene")
    assert entry["attributes"] == {"k": "v"}


def test_zip_writer_write_records_attributes(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with ZipDatasetWriter(zip_path, head=sample_head(name="Aug", modality="rgb")) as writer:
        writer.write(
            "/scene/0001", "0001.png", b"data", attributes={"weight": 0.42},
        )
        writer.save_index()

    loaded = index_dataset_from_path(zip_path)
    entry = _first_file_entry(loaded, "scene")
    assert entry["attributes"] == {"weight": 0.42}


def test_attributes_survive_index_save_load_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Aug", modality="rgb"))
    writer.get_path(
        "/scene/0001", "0001.png",
        attributes={"nested": {"a": 1}, "list": [1, 2, 3], "s": "ok"},
    ).write_bytes(b"data")
    writer.save_index()

    loaded = index_dataset_from_path(root)
    entry = _first_file_entry(loaded, "scene")
    assert entry["attributes"] == {"nested": {"a": 1}, "list": [1, 2, 3], "s": "ok"}
