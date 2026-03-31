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
