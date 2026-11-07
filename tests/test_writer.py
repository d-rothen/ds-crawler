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


def test_dataset_writer_get_path_uses_hierarchy_values_and_source_entry(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Segmentation", modality="rgb"))
    path = writer.get_path(
        "/scene:Scene01/cam:Cam0/0001",
        "0001.png",
        source_entry={
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


def test_set_modality_meta_replaces_the_saved_meta(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="Cropped", modality="rgb"))
    meta = writer.head["modality"]["meta"]
    meta["crop"] = {"top": 2, "left": 3, "height": 4, "width": 5}
    writer.set_modality_meta(meta)
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    saved = index_dataset_from_path(root)["head"]["modality"]["meta"]
    assert saved["crop"] == {"top": 2, "left": 3, "height": 4, "width": 5}
    assert saved["range"] == sample_head(modality="rgb")["modality"]["meta"]["range"]


def test_set_modality_meta_is_validated_like_the_head(tmp_path: Path) -> None:
    import pytest

    writer = DatasetWriter(tmp_path / "dataset", head=sample_head(modality="rgb"))
    with pytest.raises(ValueError, match="range"):
        writer.set_modality_meta({})
    with pytest.raises(TypeError, match="mapping"):
        writer.set_modality_meta(None)  # type: ignore[arg-type]


def test_path_prefix_keeps_scoped_datasets_apart_in_one_root(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    for scope in ("rgb", "rgb_copy"):
        writer = DatasetWriter(
            root,
            head=sample_head(name=scope, modality="rgb"),
            metadata_scope=scope,
            path_prefix=scope,
        )
        assert writer.path_prefix == scope
        path = writer.get_path("/scene/0001", "0001.png")
        path.write_bytes(scope.encode())
        writer.save_index()

    for scope in ("rgb", "rgb_copy"):
        loaded = index_dataset_from_path(root, metadata_scope=scope)
        files = loaded["index"]["children"]["scene"]["files"]
        assert [entry["path"] for entry in files] == [f"{scope}/scene/0001.png"]
        assert (root / scope / "scene" / "0001.png").read_bytes() == scope.encode()


def test_zip_writer_path_prefix_prefixes_entry_names(tmp_path: Path) -> None:
    import zipfile

    zip_path = tmp_path / "bundle.zip"
    with ZipDatasetWriter(zip_path, head=sample_head(modality="rgb"), path_prefix="rgb/") as writer:
        writer.write("/0001", "0001.png", b"data")
        writer.save_index()
    with zipfile.ZipFile(zip_path) as archive:
        assert "rgb/0001.png" in archive.namelist()


def test_path_prefix_must_stay_inside_the_root(tmp_path: Path) -> None:
    import pytest

    for prefix in ("../outside", "/absolute", "a/../../b"):
        with pytest.raises(ValueError, match="path_prefix"):
            DatasetWriter(tmp_path / "dataset", head=sample_head(modality="rgb"), path_prefix=prefix)
    writer = DatasetWriter(tmp_path / "dataset", head=sample_head(modality="rgb"), path_prefix="./")
    assert writer.path_prefix is None
