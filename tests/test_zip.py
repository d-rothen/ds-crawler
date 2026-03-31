from __future__ import annotations

from pathlib import Path

from ds_crawler import DatasetWriter, copy_dataset, index_dataset_from_path
from ds_crawler.validation import validate_dataset

from .current_helpers import sample_head


def test_index_dataset_from_path_loads_saved_zip_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    writer = DatasetWriter(source, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    zip_path = tmp_path / "dataset.zip"
    copy_dataset(source, zip_path)
    loaded = index_dataset_from_path(zip_path)

    assert loaded["head"]["dataset"]["name"] == "RGB"
    assert loaded["index"]["children"]["scene"]["files"][0]["path"] == "scene/0001.png"


def test_copy_dataset_from_zip_to_directory_preserves_metadata(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    writer = DatasetWriter(source, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    zip_path = tmp_path / "dataset.zip"
    copy_dataset(source, zip_path)

    target = tmp_path / "copied"
    copy_dataset(zip_path, target)
    validated = validate_dataset(target)

    assert validated["has_head"] is True
    assert validated["output"]["head"]["dataset"]["name"] == "RGB"
