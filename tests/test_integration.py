from __future__ import annotations

from pathlib import Path

from ds_crawler import DatasetWriter, copy_dataset, create_dataset_splits, index_dataset_from_path
from ds_crawler.validation import validate_output

from .current_helpers import sample_head


def test_end_to_end_directory_to_zip_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
    writer.save_index()
    create_dataset_splits(source, ["train"], [1.0], seed=4)

    target_zip = tmp_path / "dataset.zip"
    copy_dataset(source, target_zip)
    loaded = index_dataset_from_path(target_zip)

    assert loaded["head"]["dataset"]["name"] == "RGB"
    assert loaded["head"]["modality"]["meta"]["file_types"] == ["png"]
    assert validate_output(loaded)["contract"]["kind"] == "dataset_index"
