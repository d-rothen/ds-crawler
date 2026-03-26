"""Tests for inline split metadata stored under ``.ds_crawler/``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_crawler import (
    align_datasets,
    create_aligned_dataset_splits,
    create_dataset_splits,
    list_dataset_splits,
    load_dataset_split,
)
from ds_crawler.parser import index_dataset
from ds_crawler.traversal import (
    _collect_file_entries_by_id,
    collect_qualified_ids,
    filter_index_by_qualified_ids,
)
from ds_crawler.zip_utils import read_json_from_zip

from .conftest import (
    create_depth_predictions_tree,
    make_depth_predictions_config,
)


def test_create_dataset_splits_writes_dataset_only_payloads(
    depth_predictions_root: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    full_index = index_dataset(cfg, save_index=True)

    result = create_dataset_splits(
        depth_predictions_root,
        ["train", "val"],
        [50, 50],
        seed=42,
    )

    assert list_dataset_splits(depth_predictions_root) == ["train", "val"]

    train_path = depth_predictions_root / ".ds_crawler" / "split_train.json"
    assert train_path.is_file()
    with open(train_path) as f:
        train_payload = json.load(f)

    expected = filter_index_by_qualified_ids(
        full_index,
        result["qualified_id_splits"][0],
    )["dataset"]
    assert train_payload == expected
    assert "name" not in train_payload
    assert result["splits"][0]["metadata_file"] == ".ds_crawler/split_train.json"


def test_load_dataset_split_returns_full_output(
    depth_predictions_root: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    full_index = index_dataset(cfg, save_index=True)
    result = create_dataset_splits(
        depth_predictions_root,
        ["train", "val"],
        [50, 50],
        seed=7,
    )

    loaded = load_dataset_split(depth_predictions_root, "train")
    expected = filter_index_by_qualified_ids(
        full_index,
        result["qualified_id_splits"][0],
    )

    assert loaded["name"] == full_index["name"]
    assert loaded["type"] == full_index["type"]
    assert loaded["dataset"] == expected["dataset"]


def test_create_dataset_splits_supports_sampling_and_float_ratios(
    depth_predictions_root: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    full_index = index_dataset(cfg, save_index=True)

    result = create_dataset_splits(
        depth_predictions_root,
        ["subset"],
        [0.5],
        sample=2,
    )

    expected_selected = set(sorted(collect_qualified_ids(full_index))[::2])
    assigned = result["qualified_id_splits"][0]

    assert result["selected_qualified_ids"] == expected_selected
    assert assigned | result["unassigned_qualified_ids"] == expected_selected


def test_create_dataset_splits_indexes_and_saves_output_when_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "depth_predictions"
    create_depth_predictions_tree(root)
    cfg = make_depth_predictions_config(str(root))

    with open(root / "ds-crawler.json", "w") as f:
        json.dump(cfg, f, indent=2)

    result = create_dataset_splits(root, ["train", "val"], [50, 50], seed=1)

    assert result["total_ids"] > 0
    assert (root / ".ds_crawler" / "output.json").is_file()
    assert (root / ".ds_crawler" / "split_train.json").is_file()
    assert (root / ".ds_crawler" / "split_val.json").is_file()


def test_create_dataset_splits_rejects_invalid_or_duplicate_names(
    depth_predictions_root: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    index_dataset(cfg, save_index=True)

    with pytest.raises(ValueError, match="split_names must be unique"):
        create_dataset_splits(
            depth_predictions_root,
            ["train", "train"],
            [50, 50],
        )

    with pytest.raises(ValueError, match="split_name"):
        create_dataset_splits(
            depth_predictions_root,
            ["bad/name", "val"],
            [50, 50],
        )


def test_create_aligned_dataset_splits_and_align_by_split(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "depth_a"
    root_b = tmp_path / "depth_b"
    create_depth_predictions_tree(root_a)
    create_depth_predictions_tree(root_b)

    cfg_a = make_depth_predictions_config(str(root_a))
    cfg_b = make_depth_predictions_config(str(root_b))
    cfg_b["name"] = "depth_predictions_b"

    index_dataset(cfg_a, save_index=True)
    index_dataset(cfg_b, save_index=True)

    result = create_aligned_dataset_splits(
        [root_a, root_b],
        ["train", "val"],
        [50, 50],
        seed=3,
    )

    train_ids = result["qualified_id_splits"][0]
    loaded_a = load_dataset_split(root_a, "train")
    loaded_b = load_dataset_split(root_b, "train")

    assert collect_qualified_ids(loaded_a) == train_ids
    assert collect_qualified_ids(loaded_b) == train_ids

    aligned = align_datasets(
        {"modality": "a", "source": root_a, "split": "train"},
        {"modality": "b", "source": root_b, "split": "train"},
    )

    expected_ids = set(_collect_file_entries_by_id(loaded_a["dataset"]).keys())
    assert set(aligned) == expected_ids
    assert all(set(modalities) == {"a", "b"} for modalities in aligned.values())


def test_inline_splits_work_for_prefixed_zip(
    depth_predictions_prefixed_zip: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_prefixed_zip))
    full_index = index_dataset(cfg, save_index=True)

    result = create_dataset_splits(
        depth_predictions_prefixed_zip,
        ["train", "val"],
        [50, 50],
        seed=11,
    )

    assert list_dataset_splits(depth_predictions_prefixed_zip) == ["train", "val"]

    train_payload = read_json_from_zip(
        depth_predictions_prefixed_zip,
        ".ds_crawler/split_train.json",
    )
    assert train_payload is not None
    assert "name" not in train_payload

    loaded = load_dataset_split(depth_predictions_prefixed_zip, "train")
    expected = filter_index_by_qualified_ids(
        full_index,
        result["qualified_id_splits"][0],
    )
    assert loaded["dataset"] == expected["dataset"]


def test_load_dataset_split_raises_when_missing(
    depth_predictions_root: Path,
) -> None:
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    index_dataset(cfg, save_index=True)

    with pytest.raises(FileNotFoundError, match="split_train.json"):
        load_dataset_split(depth_predictions_root, "train")
