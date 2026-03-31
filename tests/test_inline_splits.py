from __future__ import annotations

from pathlib import Path

from ds_crawler import (
    DatasetWriter,
    create_aligned_dataset_splits,
    create_dataset_splits,
    list_dataset_splits,
    load_dataset_split,
)
from ds_crawler.traversal import collect_qualified_ids

from .current_helpers import sample_head


def test_create_dataset_splits_lists_and_loads_split(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
    writer.get_path("/scene/0003", "0003.png").write_bytes(b"data")
    writer.save_index()

    create_dataset_splits(root, ["train", "val"], [50, 50], seed=1)
    loaded = load_dataset_split(root, "train")

    assert list_dataset_splits(root) == ["train", "val"]
    assert loaded["split"]["name"] == "train"
    assert loaded["split"]["source_index_file"] == "index.json"


def test_create_aligned_dataset_splits_reuses_same_partition(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root, modality in ((root_a, "rgb"), (root_b, "depth")):
        writer = DatasetWriter(root, head=sample_head(name=root.name, modality=modality))
        writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
        writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
        writer.save_index()

    result = create_aligned_dataset_splits(
        [root_a, root_b],
        ["train", "val"],
        [50, 50],
        seed=2,
    )

    loaded_a = load_dataset_split(root_a, "train")
    loaded_b = load_dataset_split(root_b, "train")

    assert collect_qualified_ids(loaded_a) == result["qualified_id_splits"][0]
    assert collect_qualified_ids(loaded_b) == result["qualified_id_splits"][0]
