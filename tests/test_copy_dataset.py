from __future__ import annotations

from pathlib import Path

from ds_crawler import DatasetWriter, copy_dataset, get_files, index_dataset_from_path
from ds_crawler.zip_utils import OUTPUT_FILENAME, read_metadata_json

from .current_helpers import sample_head


def test_copy_dataset_to_directory_writes_canonical_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="Source RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
    writer.save_index()

    target = tmp_path / "copied"
    result = copy_dataset(source, target)
    loaded = index_dataset_from_path(target)

    assert result["copied"] == 2
    assert (target / ".ds_crawler" / "dataset-head.json").is_file()
    assert (target / ".ds_crawler" / "ds-crawler.json").is_file()
    assert (target / ".ds_crawler" / OUTPUT_FILENAME).is_file()
    assert set(get_files(loaded)) == {"scene/0001.png", "scene/0002.png"}


def test_copy_dataset_to_zip_writes_index_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="Source RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.save_index()

    target_zip = tmp_path / "copied.zip"
    copy_dataset(source, target_zip)

    index_artifact = read_metadata_json(target_zip, OUTPUT_FILENAME)
    loaded = index_dataset_from_path(target_zip)

    assert index_artifact is not None
    assert index_artifact["contract"]["kind"] == "dataset_index"
    assert "head" not in index_artifact
    assert loaded["head"]["dataset"]["name"] == "Source RGB"


def test_copy_dataset_reports_missing_referenced_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="Source RGB"))
    missing_path = writer.get_path("/scene/0001", "0001.png")
    kept_path = writer.get_path("/scene/0002", "0002.png")
    missing_path.write_bytes(b"gone soon")
    kept_path.write_bytes(b"keep")
    writer.save_index()
    missing_path.unlink()

    target = tmp_path / "copied"
    result = copy_dataset(source, target)

    assert result["copied"] == 1
    assert result["missing"] == 1
    assert result["missing_files"] == ["scene/0001.png"]
