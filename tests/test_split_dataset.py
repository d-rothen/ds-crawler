from __future__ import annotations

from pathlib import Path

from ds_crawler import DatasetWriter, index_dataset_from_path, split_dataset, split_datasets
from ds_crawler.traversal import collect_qualified_ids

from .current_helpers import sample_head


def test_split_dataset_writes_disjoint_target_indices(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="RGB"))
    for frame in ("0001", "0002", "0003", "0004"):
        writer.get_path(f"/scene/{frame}", f"{frame}.png").write_bytes(b"data")
    writer.save_index()

    train = tmp_path / "train"
    val = tmp_path / "val"
    split_dataset(source, [50, 50], [train, val], seed=1)

    train_ids = collect_qualified_ids(index_dataset_from_path(train))
    val_ids = collect_qualified_ids(index_dataset_from_path(val))

    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids | val_ids) == 4


def test_split_datasets_uses_common_id_intersection(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"

    writer_a = DatasetWriter(root_a, head=sample_head(name="A"))
    for frame in ("0001", "0002", "0003"):
        writer_a.get_path(f"/scene/{frame}", f"{frame}.png").write_bytes(b"data")
    writer_a.save_index()

    writer_b = DatasetWriter(root_b, head=sample_head(name="B", modality="depth"))
    for frame in ("0001", "0002"):
        writer_b.get_path(f"/scene/{frame}", f"{frame}.png").write_bytes(b"data")
    writer_b.save_index()

    result = split_datasets([root_a, root_b], ["train", "val"], [50, 50], seed=2)

    train_a = index_dataset_from_path(tmp_path / "a_train")
    train_b = index_dataset_from_path(tmp_path / "b_train")

    assert result["per_source"][0]["excluded_ids"] == 1
    assert collect_qualified_ids(train_a) == collect_qualified_ids(train_b)
