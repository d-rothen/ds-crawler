"""Tests for split_dataset, filter_index_by_qualified_ids, and related functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.parser import index_dataset
from ds_crawler.traversal import (
    collect_qualified_ids,
    filter_index_by_qualified_ids,
    get_files,
    split_qualified_ids,
)
from ds_crawler.operations import (
    copy_dataset,
    split_dataset,
    split_datasets,
    _derive_split_path,
)

from .conftest import (
    create_depth_predictions_tree,
    create_vkitti2_tree,
    make_depth_predictions_config,
    make_vkitti2_config,
    touch,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic indices
# ---------------------------------------------------------------------------


def _make_flat_index(ids: list[str]) -> dict[str, Any]:
    """Build a minimal index with no hierarchy (files at dataset root)."""
    return {
        "name": "flat",
        "dataset": {
            "files": [
                {"path": f"{fid}.png", "id": fid}
                for fid in ids
            ]
        },
    }


def _make_hierarchical_index(
    scenes: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a synthetic hierarchical index.

    *scenes* maps a hierarchy key to a list of file IDs, e.g.
    ``{"scene:A": ["001", "002"], "scene:B": ["001", "003"]}``.

    File paths use a filesystem-safe version of the key (colons replaced
    by underscores) so that real files can be created at those paths when
    needed.
    """
    children = {}
    for scene_key, ids in scenes.items():
        dir_name = scene_key.replace(":", "_")
        children[scene_key] = {
            "files": [
                {"path": f"{dir_name}/{fid}.png", "id": fid}
                for fid in ids
            ]
        }
    return {
        "name": "hierarchical",
        "dataset": {"children": children},
    }


def _make_nested_hierarchical_index(
    tree: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """Build a deeper hierarchy: scene → camera → file IDs."""
    scene_children: dict[str, Any] = {}
    for scene_key, cameras in tree.items():
        scene_dir = scene_key.replace(":", "_")
        cam_children: dict[str, Any] = {}
        for cam_key, ids in cameras.items():
            cam_dir = cam_key.replace(":", "_")
            cam_children[cam_key] = {
                "files": [
                    {"path": f"{scene_dir}/{cam_dir}/{fid}.png", "id": fid}
                    for fid in ids
                ]
            }
        scene_children[scene_key] = {"children": cam_children}
    return {
        "name": "nested",
        "dataset": {"children": scene_children},
    }


# ---------------------------------------------------------------------------
# TestCollectQualifiedIds
# ---------------------------------------------------------------------------


class TestCollectQualifiedIds:
    """Tests for collect_qualified_ids."""

    def test_flat_index(self) -> None:
        index = _make_flat_index(["a", "b", "c"])
        qids = collect_qualified_ids(index)
        assert qids == {("a",), ("b",), ("c",)}

    def test_hierarchical_index(self) -> None:
        index = _make_hierarchical_index({
            "scene:A": ["001", "002"],
            "scene:B": ["001", "003"],
        })
        qids = collect_qualified_ids(index)
        expected = {
            ("scene:A", "001"),
            ("scene:A", "002"),
            ("scene:B", "001"),
            ("scene:B", "003"),
        }
        assert qids == expected

    def test_same_id_different_hierarchy_distinguished(self) -> None:
        """ID '001' under scene:A and scene:B are separate qualified IDs."""
        index = _make_hierarchical_index({
            "scene:A": ["001"],
            "scene:B": ["001"],
        })
        qids = collect_qualified_ids(index)
        assert len(qids) == 2
        assert ("scene:A", "001") in qids
        assert ("scene:B", "001") in qids

    def test_nested_hierarchy(self) -> None:
        index = _make_nested_hierarchical_index({
            "scene:A": {"cam:0": ["f1", "f2"], "cam:1": ["f1"]},
        })
        qids = collect_qualified_ids(index)
        expected = {
            ("scene:A", "cam:0", "f1"),
            ("scene:A", "cam:0", "f2"),
            ("scene:A", "cam:1", "f1"),
        }
        assert qids == expected

    def test_empty_index(self) -> None:
        index: dict[str, Any] = {"name": "empty", "dataset": {}}
        qids = collect_qualified_ids(index)
        assert qids == set()

    def test_with_real_vkitti2(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        idx = index_dataset(cfg)
        qids = collect_qualified_ids(idx)
        # VKITTI2 mock has 4 data frames
        assert len(qids) == 4
        # Each qualified ID: 4 hierarchy levels + 1 file id
        for qid in qids:
            assert len(qid) == 5


# ---------------------------------------------------------------------------
# TestFilterIndexByQualifiedIds
# ---------------------------------------------------------------------------


class TestFilterIndexByQualifiedIds:
    """Tests for filter_index_by_qualified_ids."""

    def test_keeps_matching_entries(self) -> None:
        index = _make_hierarchical_index({
            "scene:A": ["001", "002"],
            "scene:B": ["003"],
        })
        keep = {("scene:A", "001"), ("scene:B", "003")}
        result = filter_index_by_qualified_ids(index, keep)

        files_a = result["dataset"]["children"]["scene:A"]["files"]
        files_b = result["dataset"]["children"]["scene:B"]["files"]
        assert len(files_a) == 1
        assert files_a[0]["id"] == "001"
        assert len(files_b) == 1
        assert files_b[0]["id"] == "003"

    def test_removes_non_matching_entries(self) -> None:
        index = _make_hierarchical_index({"scene:A": ["001", "002"]})
        keep = {("scene:A", "001")}
        result = filter_index_by_qualified_ids(index, keep)

        files = result["dataset"]["children"]["scene:A"]["files"]
        assert len(files) == 1
        assert files[0]["id"] == "001"

    def test_empty_qualified_ids_removes_all_files(self) -> None:
        index = _make_hierarchical_index({"scene:A": ["001", "002"]})
        result = filter_index_by_qualified_ids(index, set())
        files = result["dataset"]["children"]["scene:A"]["files"]
        assert files == []

    def test_preserves_camera_metadata(self) -> None:
        """Camera intrinsics/extrinsics on hierarchy nodes are preserved."""
        index: dict[str, Any] = {
            "name": "test",
            "id_regex": "some_regex",
            "dataset": {
                "children": {
                    "scene:A": {
                        "camera_intrinsics": "A/intrinsics.txt",
                        "camera_extrinsics": "A/extrinsics.txt",
                        "files": [
                            {"path": "A/001.png", "id": "001"},
                            {"path": "A/002.png", "id": "002"},
                        ],
                    }
                }
            },
        }
        keep = {("scene:A", "001")}
        result = filter_index_by_qualified_ids(index, keep)

        node = result["dataset"]["children"]["scene:A"]
        assert node["camera_intrinsics"] == "A/intrinsics.txt"
        assert node["camera_extrinsics"] == "A/extrinsics.txt"
        assert len(node["files"]) == 1

    def test_preserves_top_level_metadata(self) -> None:
        index: dict[str, Any] = {
            "name": "test",
            "id_regex": "pattern",
            "some_prop": "value",
            "dataset": {"files": [{"path": "a.png", "id": "a"}]},
        }
        result = filter_index_by_qualified_ids(index, {("a",)})
        assert result["name"] == "test"
        assert result["id_regex"] == "pattern"
        assert result["some_prop"] == "value"

    def test_hierarchy_distinguishes_same_id(self) -> None:
        """Filtering for scene:A/001 does NOT include scene:B/001."""
        index = _make_hierarchical_index({
            "scene:A": ["001"],
            "scene:B": ["001"],
        })
        keep = {("scene:A", "001")}
        result = filter_index_by_qualified_ids(index, keep)

        files_a = result["dataset"]["children"]["scene:A"]["files"]
        files_b = result["dataset"]["children"]["scene:B"]["files"]
        assert len(files_a) == 1
        assert len(files_b) == 0

    def test_nested_hierarchy_filtering(self) -> None:
        index = _make_nested_hierarchical_index({
            "scene:A": {"cam:0": ["f1", "f2"], "cam:1": ["f1"]},
        })
        keep = {("scene:A", "cam:0", "f1"), ("scene:A", "cam:1", "f1")}
        result = filter_index_by_qualified_ids(index, keep)

        cam0 = result["dataset"]["children"]["scene:A"]["children"]["cam:0"]
        cam1 = result["dataset"]["children"]["scene:A"]["children"]["cam:1"]
        assert len(cam0["files"]) == 1  # f1 kept, f2 removed
        assert cam0["files"][0]["id"] == "f1"
        assert len(cam1["files"]) == 1  # f1 kept

    def test_roundtrip_collect_then_filter_is_identity(self) -> None:
        """Filtering with all collected IDs preserves every file entry."""
        index = _make_hierarchical_index({
            "scene:A": ["001", "002"],
            "scene:B": ["003"],
        })
        all_ids = collect_qualified_ids(index)
        result = filter_index_by_qualified_ids(index, all_ids)
        assert get_files(result) == get_files(index)


# ---------------------------------------------------------------------------
# TestSplitQualifiedIds
# ---------------------------------------------------------------------------


class TestSplitQualifiedIds:
    """Tests for split_qualified_ids."""

    def test_basic_two_way_split(self) -> None:
        ids = {(str(i),) for i in range(100)}
        splits = split_qualified_ids(ids, [70, 30])

        assert len(splits) == 2
        assert len(splits[0]) == 70
        assert len(splits[1]) == 30
        assert splits[0] | splits[1] == ids
        assert splits[0] & splits[1] == set()

    def test_three_way_split(self) -> None:
        ids = {(str(i),) for i in range(100)}
        splits = split_qualified_ids(ids, [70, 20, 10])

        assert len(splits) == 3
        assert len(splits[0]) == 70
        assert len(splits[1]) == 20
        assert len(splits[2]) == 10
        # Disjoint and complete
        all_assigned: set[tuple[str, ...]] = set()
        for s in splits:
            assert all_assigned & s == set()
            all_assigned |= s
        assert all_assigned == ids

    def test_deterministic_without_seed(self) -> None:
        ids = {("scene:A", str(i)) for i in range(50)}
        s1 = split_qualified_ids(ids, [60, 40])
        s2 = split_qualified_ids(ids, [60, 40])
        assert s1 == s2

    def test_deterministic_with_seed(self) -> None:
        ids = {("scene:A", str(i)) for i in range(50)}
        s1 = split_qualified_ids(ids, [60, 40], seed=42)
        s2 = split_qualified_ids(ids, [60, 40], seed=42)
        assert s1 == s2

    def test_different_seeds_give_different_splits(self) -> None:
        ids = {(str(i),) for i in range(100)}
        s1 = split_qualified_ids(ids, [50, 50], seed=1)
        s2 = split_qualified_ids(ids, [50, 50], seed=2)
        assert s1[0] != s2[0]

    def test_seed_shuffles_relative_to_sorted(self) -> None:
        ids = {(str(i),) for i in range(100)}
        no_seed = split_qualified_ids(ids, [50, 50])
        with_seed = split_qualified_ids(ids, [50, 50], seed=42)
        assert no_seed[0] != with_seed[0]

    def test_rounding_preserves_all_ids(self) -> None:
        """No IDs are lost when ratios don't divide evenly."""
        ids = {(str(i),) for i in range(101)}
        splits = split_qualified_ids(ids, [70, 30])
        assert splits[0] | splits[1] == ids
        assert len(splits[0]) + len(splits[1]) == 101

    def test_empty_set(self) -> None:
        splits = split_qualified_ids(set(), [50, 50])
        assert len(splits) == 2
        assert splits[0] == set()
        assert splits[1] == set()

    def test_single_element(self) -> None:
        ids = {("only",)}
        splits = split_qualified_ids(ids, [90, 10])
        assert splits[0] | splits[1] == ids

    def test_single_100_percent_split(self) -> None:
        ids = {(str(i),) for i in range(10)}
        splits = split_qualified_ids(ids, [100])
        assert len(splits) == 1
        assert splits[0] == ids

    def test_raises_on_sum_not_100(self) -> None:
        with pytest.raises(ValueError, match="sum to 100"):
            split_qualified_ids(set(), [50, 40])

    def test_raises_on_empty_ratios(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            split_qualified_ids(set(), [])

    def test_raises_on_zero_ratio(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            split_qualified_ids(set(), [100, 0])

    def test_raises_on_negative_ratio(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            split_qualified_ids(set(), [110, -10])


# ---------------------------------------------------------------------------
# TestSplitDataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    """End-to-end tests for split_dataset."""

    @staticmethod
    def _prepare_indexed_dataset(
        root: Path, create_tree, make_config,
    ) -> dict[str, Any]:
        """Create tree, index it, write output.json, return the index."""
        create_tree(root)
        cfg = make_config(str(root))
        idx = index_dataset(cfg)
        with open(root / "output.json", "w") as f:
            json.dump(idx, f)
        return idx

    def test_basic_split(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        idx = self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        all_qids = collect_qualified_ids(idx)

        dst_a = tmp_path / "split_a"
        dst_b = tmp_path / "split_b"
        result = split_dataset(root, [50, 50], [dst_a, dst_b])

        # All IDs partitioned without overlap
        split_a_ids = result["qualified_id_splits"][0]
        split_b_ids = result["qualified_id_splits"][1]
        assert split_a_ids | split_b_ids == all_qids
        assert split_a_ids & split_b_ids == set()

        # No missing files
        for s in result["splits"]:
            assert s["missing"] == 0

        # output.json written in each target
        assert (dst_a / ".ds_crawler" / "output.json").is_file()
        assert (dst_b / ".ds_crawler" / "output.json").is_file()

    def test_three_way_split(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        targets = [tmp_path / f"split_{i}" for i in range(3)]
        result = split_dataset(root, [50, 30, 20], targets)

        assert len(result["splits"]) == 3
        assert len(result["qualified_id_splits"]) == 3
        # Disjoint partition
        all_ids: set[tuple[str, ...]] = set()
        for split_ids in result["qualified_id_splits"]:
            assert all_ids & split_ids == set()
            all_ids |= split_ids

    def test_result_contains_ratio_and_target(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        dst_train = tmp_path / "train"
        dst_val = tmp_path / "val"
        result = split_dataset(root, [80, 20], [dst_train, dst_val])

        assert result["splits"][0]["ratio"] == 80
        assert result["splits"][1]["ratio"] == 20
        assert result["splits"][0]["target"] == str(dst_train)
        assert result["splits"][1]["target"] == str(dst_val)

    def test_raises_on_missing_output_json(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="output.json"):
            split_dataset(root, [50, 50], [tmp_path / "a", tmp_path / "b"])

    def test_raises_on_mismatched_lengths(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="same length"):
            split_dataset(tmp_path, [50, 50], [tmp_path / "a"])

    def test_seed_deterministic(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        r1 = split_dataset(
            root, [60, 40],
            [tmp_path / "a1", tmp_path / "b1"], seed=42,
        )
        r2 = split_dataset(
            root, [60, 40],
            [tmp_path / "a2", tmp_path / "b2"], seed=42,
        )
        assert r1["qualified_id_splits"] == r2["qualified_id_splits"]

    def test_qualified_ids_parameter_restricts_split(self, tmp_path: Path) -> None:
        """When qualified_ids is provided, only those IDs are split."""
        root = tmp_path / "src"
        idx = self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        all_qids = collect_qualified_ids(idx)
        subset = set(sorted(all_qids)[:2])

        dst_a = tmp_path / "a"
        dst_b = tmp_path / "b"
        result = split_dataset(
            root, [50, 50], [dst_a, dst_b], qualified_ids=subset,
        )
        total_split_ids = (
            result["qualified_id_splits"][0] | result["qualified_id_splits"][1]
        )
        assert total_split_ids == subset

    def test_split_preserves_camera_files(self, tmp_path: Path) -> None:
        """Camera intrinsics/extrinsics are copied in each split."""
        root = tmp_path / "src"
        self._prepare_indexed_dataset(
            root, create_vkitti2_tree, make_vkitti2_config,
        )
        dst_a = tmp_path / "train"
        dst_b = tmp_path / "val"
        result = split_dataset(root, [50, 50], [dst_a, dst_b])

        for s in result["splits"]:
            assert s["missing"] == 0

    def test_each_split_output_json_matches_files_on_disk(
        self, tmp_path: Path,
    ) -> None:
        """output.json in each split only references files actually copied."""
        root = tmp_path / "src"
        self._prepare_indexed_dataset(
            root, create_depth_predictions_tree, make_depth_predictions_config,
        )
        dst_a = tmp_path / "a"
        dst_b = tmp_path / "b"
        split_dataset(root, [50, 50], [dst_a, dst_b])

        for dst in [dst_a, dst_b]:
            with open(dst / ".ds_crawler" / "output.json") as f:
                split_idx = json.load(f)
            for rel_path in get_files(split_idx):
                assert (dst / rel_path).is_file()


# ---------------------------------------------------------------------------
# TestCrossDatasetIntersection
# ---------------------------------------------------------------------------


class TestCrossDatasetIntersection:
    """Tests for the intersection workflow across multiple aligned datasets.

    Simulates the use case of splitting aligned datasets (e.g. rgb and
    depth from the same capture session) that share a hierarchy structure
    but may have different subsets of IDs.
    """

    @staticmethod
    def _build_two_aligned_datasets(
        tmp_path: Path,
    ) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
        """Create two datasets with partially overlapping qualified IDs.

        rgb has:   scene:A/{001,002,003}, scene:B/{001}
        depth has: scene:A/{001,002},     scene:B/{001,002}
        common:    scene:A/{001,002},     scene:B/{001}
        """
        rgb_root = tmp_path / "rgb"
        rgb_root.mkdir()
        rgb_index = _make_hierarchical_index({
            "scene:A": ["001", "002", "003"],
            "scene:B": ["001"],
        })
        for f in get_files(rgb_index):
            touch(rgb_root / f)
        with open(rgb_root / "output.json", "w") as f:
            json.dump(rgb_index, f)

        depth_root = tmp_path / "depth"
        depth_root.mkdir()
        depth_index = _make_hierarchical_index({
            "scene:A": ["001", "002"],
            "scene:B": ["001", "002"],
        })
        for f in get_files(depth_index):
            touch(depth_root / f)
        with open(depth_root / "output.json", "w") as f:
            json.dump(depth_index, f)

        return rgb_root, depth_root, rgb_index, depth_index

    def test_intersection_finds_common_ids(self, tmp_path: Path) -> None:
        _, _, rgb_idx, depth_idx = self._build_two_aligned_datasets(tmp_path)

        rgb_qids = collect_qualified_ids(rgb_idx)
        depth_qids = collect_qualified_ids(depth_idx)
        common = rgb_qids & depth_qids

        expected = {
            ("scene:A", "001"),
            ("scene:A", "002"),
            ("scene:B", "001"),
        }
        assert common == expected

    def test_intersection_excludes_non_common_ids(self, tmp_path: Path) -> None:
        _, _, rgb_idx, depth_idx = self._build_two_aligned_datasets(tmp_path)

        common = collect_qualified_ids(rgb_idx) & collect_qualified_ids(depth_idx)
        # rgb-only: scene:A/003
        assert ("scene:A", "003") not in common
        # depth-only: scene:B/002
        assert ("scene:B", "002") not in common

    def test_split_both_datasets_with_same_partition(
        self, tmp_path: Path,
    ) -> None:
        """Split rgb and depth using intersected IDs; verify identical splits."""
        rgb_root, depth_root, rgb_idx, depth_idx = (
            self._build_two_aligned_datasets(tmp_path)
        )
        common = collect_qualified_ids(rgb_idx) & collect_qualified_ids(depth_idx)
        assert len(common) == 3

        # Split rgb first
        rgb_result = split_dataset(
            rgb_root, [70, 30],
            [tmp_path / "rgb_train", tmp_path / "rgb_val"],
            qualified_ids=common, seed=7,
        )

        # Apply the same ID partition to depth via filter + copy
        id_splits = rgb_result["qualified_id_splits"]
        for split_ids, name in zip(id_splits, ["depth_train", "depth_val"]):
            target = tmp_path / name
            filtered = filter_index_by_qualified_ids(depth_idx, split_ids)
            copy_dataset(depth_root, target, index=filtered)

        # Verify rgb and depth train splits have the same qualified IDs
        with open(tmp_path / "rgb_train" / ".ds_crawler" / "output.json") as f:
            rgb_train_idx = json.load(f)
        with open(tmp_path / "depth_train" / ".ds_crawler" / "output.json") as f:
            depth_train_idx = json.load(f)

        assert (
            collect_qualified_ids(rgb_train_idx)
            == collect_qualified_ids(depth_train_idx)
        )

        # Same for val
        with open(tmp_path / "rgb_val" / ".ds_crawler" / "output.json") as f:
            rgb_val_idx = json.load(f)
        with open(tmp_path / "depth_val" / ".ds_crawler" / "output.json") as f:
            depth_val_idx = json.load(f)

        assert (
            collect_qualified_ids(rgb_val_idx)
            == collect_qualified_ids(depth_val_idx)
        )

    def test_split_partition_is_disjoint_and_complete(
        self, tmp_path: Path,
    ) -> None:
        """Train + val IDs are disjoint and together cover the intersection."""
        rgb_root, _, rgb_idx, depth_idx = (
            self._build_two_aligned_datasets(tmp_path)
        )
        common = collect_qualified_ids(rgb_idx) & collect_qualified_ids(depth_idx)

        result = split_dataset(
            rgb_root, [70, 30],
            [tmp_path / "train", tmp_path / "val"],
            qualified_ids=common, seed=42,
        )
        train_ids = result["qualified_id_splits"][0]
        val_ids = result["qualified_id_splits"][1]

        assert train_ids & val_ids == set()
        assert train_ids | val_ids == common

    def test_three_way_intersection(self) -> None:
        """Intersection across three datasets."""
        idx_a = _make_hierarchical_index({"s:X": ["1", "2", "3"]})
        idx_b = _make_hierarchical_index({"s:X": ["1", "2", "4"]})
        idx_c = _make_hierarchical_index({"s:X": ["1", "3", "4"]})

        common = (
            collect_qualified_ids(idx_a)
            & collect_qualified_ids(idx_b)
            & collect_qualified_ids(idx_c)
        )
        assert common == {("s:X", "1")}

    def test_hierarchy_prevents_false_intersection(self) -> None:
        """Same file ID under different hierarchy keys must not intersect."""
        idx_a = _make_hierarchical_index({"scene:A": ["001"]})
        idx_b = _make_hierarchical_index({"scene:B": ["001"]})

        common = collect_qualified_ids(idx_a) & collect_qualified_ids(idx_b)
        assert common == set()

    def test_real_datasets_intersection_workflow(self, tmp_path: Path) -> None:
        """Full intersection workflow with real (VKITTI2) indexed datasets."""
        # First "version" — full mock VKITTI2
        vkitti_root = tmp_path / "vkitti"
        create_vkitti2_tree(vkitti_root)
        cfg = make_vkitti2_config(str(vkitti_root))
        vkitti_idx = index_dataset(cfg)
        with open(vkitti_root / "output.json", "w") as f:
            json.dump(vkitti_idx, f)

        # Second "version" — only Scene01/clone Camera_0 frames
        vkitti_root2 = tmp_path / "vkitti2"
        touch(vkitti_root2 / "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg")
        touch(vkitti_root2 / "Scene01/clone/frames/rgb/Camera_0/rgb_00002.jpg")
        touch(vkitti_root2 / "Scene01/clone/intrinsics/Camera_0_intrinsics.txt")
        touch(vkitti_root2 / "Scene01/clone/extrinsics/Camera_0_extrinsics.txt")
        cfg2 = make_vkitti2_config(str(vkitti_root2))
        vkitti_idx2 = index_dataset(cfg2)
        with open(vkitti_root2 / "output.json", "w") as f:
            json.dump(vkitti_idx2, f)

        q1 = collect_qualified_ids(vkitti_idx)
        q2 = collect_qualified_ids(vkitti_idx2)
        common = q1 & q2

        # Second dataset is a strict subset of the first
        assert len(q2) == 2
        assert common == q2

        # Split using common IDs
        result = split_dataset(
            vkitti_root, [50, 50],
            [tmp_path / "train", tmp_path / "val"],
            qualified_ids=common,
        )
        total = (
            result["qualified_id_splits"][0] | result["qualified_id_splits"][1]
        )
        assert total == common

    def test_files_on_disk_match_split_index(self, tmp_path: Path) -> None:
        """After intersection-split-copy, files on disk match output.json."""
        rgb_root, depth_root, rgb_idx, depth_idx = (
            self._build_two_aligned_datasets(tmp_path)
        )
        common = collect_qualified_ids(rgb_idx) & collect_qualified_ids(depth_idx)

        rgb_result = split_dataset(
            rgb_root, [70, 30],
            [tmp_path / "rgb_train", tmp_path / "rgb_val"],
            qualified_ids=common,
        )
        id_splits = rgb_result["qualified_id_splits"]

        # Copy depth using the same partition
        for split_ids, name in zip(id_splits, ["depth_train", "depth_val"]):
            filtered = filter_index_by_qualified_ids(depth_idx, split_ids)
            copy_dataset(depth_root, tmp_path / name, index=filtered)

        # Verify files on disk for all four targets
        for name in ["rgb_train", "rgb_val", "depth_train", "depth_val"]:
            target = tmp_path / name
            with open(target / ".ds_crawler" / "output.json") as f:
                split_idx = json.load(f)
            for rel_path in get_files(split_idx):
                assert (target / rel_path).is_file(), (
                    f"{rel_path} missing in {name}"
                )


# ---------------------------------------------------------------------------
# TestDeriveSplitPath
# ---------------------------------------------------------------------------


class TestDeriveSplitPath:
    """Tests for _derive_split_path."""

    def test_directory(self) -> None:
        assert _derive_split_path(Path("/data/kitti_rgb"), "train") == Path(
            "/data/kitti_rgb_train"
        )

    def test_zip(self) -> None:
        assert _derive_split_path(Path("/data/kitti_rgb.zip"), "train") == Path(
            "/data/kitti_rgb_train.zip"
        )

    def test_zip_case_insensitive(self) -> None:
        assert _derive_split_path(Path("/data/ds.ZIP"), "val") == Path(
            "/data/ds_val.zip"
        )

    def test_multiple_suffixes(self) -> None:
        base = Path("/data/kitti")
        assert _derive_split_path(base, "train") == Path("/data/kitti_train")
        assert _derive_split_path(base, "val") == Path("/data/kitti_val")
        assert _derive_split_path(base, "test") == Path("/data/kitti_test")


# ---------------------------------------------------------------------------
# TestSplitDatasets
# ---------------------------------------------------------------------------


class TestSplitDatasets:
    """Tests for split_datasets (multi-source intersection-based split)."""

    @staticmethod
    def _build_source(
        root: Path, scenes: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Create a source directory with files and output.json."""
        root.mkdir(parents=True, exist_ok=True)
        index = _make_hierarchical_index(scenes)
        for f in get_files(index):
            touch(root / f)
        with open(root / "output.json", "w") as f:
            json.dump(index, f)
        return index

    # -- basic behaviour --

    def test_two_sources_two_way_split(self, tmp_path: Path) -> None:
        """Split two aligned modalities into train/val."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"

        self._build_source(rgb_root, {
            "scene:A": ["001", "002", "003"],
            "scene:B": ["001"],
        })
        self._build_source(depth_root, {
            "scene:A": ["001", "002"],
            "scene:B": ["001", "002"],
        })

        result = split_datasets(
            [rgb_root, depth_root],
            ["train", "val"],
            [70, 30],
            seed=42,
        )

        # Intersection = scene:A/{001,002} + scene:B/{001} = 3 IDs
        assert len(result["common_ids"]) == 3
        # Partition is disjoint and complete
        train_ids = result["qualified_id_splits"][0]
        val_ids = result["qualified_id_splits"][1]
        assert train_ids & val_ids == set()
        assert train_ids | val_ids == result["common_ids"]

        # Two per-source entries
        assert len(result["per_source"]) == 2

    def test_target_paths_derived_correctly(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {"s:A": ["1"]})
        self._build_source(depth_root, {"s:A": ["1"]})

        result = split_datasets(
            [rgb_root, depth_root], ["train", "val"], [50, 50],
        )

        # Check derived target paths
        rgb_splits = result["per_source"][0]["splits"]
        assert rgb_splits[0]["target"] == str(tmp_path / "rgb_train")
        assert rgb_splits[1]["target"] == str(tmp_path / "rgb_val")

        depth_splits = result["per_source"][1]["splits"]
        assert depth_splits[0]["target"] == str(tmp_path / "depth_train")
        assert depth_splits[1]["target"] == str(tmp_path / "depth_val")

    def test_files_copied_to_derived_paths(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {"s:A": ["1", "2"]})
        self._build_source(depth_root, {"s:A": ["1", "2"]})

        split_datasets(
            [rgb_root, depth_root], ["train", "val"], [50, 50],
        )

        for name in ["rgb_train", "rgb_val", "depth_train", "depth_val"]:
            target = tmp_path / name
            assert (target / ".ds_crawler" / "output.json").is_file(), f"missing output.json in {name}"

    # -- intersection correctness --

    def test_intersection_excludes_non_common_ids(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {
            "scene:A": ["001", "002", "003"],
        })
        self._build_source(depth_root, {
            "scene:A": ["001", "002"],
        })

        result = split_datasets(
            [rgb_root, depth_root], ["train", "val"], [50, 50],
        )

        # "003" is rgb-only → excluded from all splits
        all_split_ids = (
            result["qualified_id_splits"][0] | result["qualified_id_splits"][1]
        )
        assert ("scene:A", "003") not in all_split_ids

    def test_hierarchy_prevents_false_intersection(self, tmp_path: Path) -> None:
        """Same ID under different hierarchy keys is not a match."""
        src_a = tmp_path / "modality_a"
        src_b = tmp_path / "modality_b"
        self._build_source(src_a, {"scene:A": ["001"]})
        self._build_source(src_b, {"scene:B": ["001"]})

        result = split_datasets(
            [src_a, src_b], ["train", "val"], [50, 50],
        )
        assert len(result["common_ids"]) == 0

    def test_three_source_intersection(self, tmp_path: Path) -> None:
        src_a = tmp_path / "a"
        src_b = tmp_path / "b"
        src_c = tmp_path / "c"
        self._build_source(src_a, {"s:X": ["1", "2", "3"]})
        self._build_source(src_b, {"s:X": ["1", "2", "4"]})
        self._build_source(src_c, {"s:X": ["1", "3", "4"]})

        result = split_datasets(
            [src_a, src_b, src_c], ["train", "val"], [50, 50],
        )
        assert result["common_ids"] == {("s:X", "1")}

    # -- cross-modality correspondence --

    def test_same_qualified_ids_in_each_split_across_sources(
        self, tmp_path: Path,
    ) -> None:
        """Every modality's train split has the same qualified IDs."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        normals_root = tmp_path / "normals"

        common_scenes = {"s:A": ["1", "2", "3", "4"], "s:B": ["1", "2"]}
        self._build_source(rgb_root, {
            **common_scenes, "s:C": ["extra"],
        })
        self._build_source(depth_root, common_scenes)
        self._build_source(normals_root, {
            **common_scenes, "s:D": ["also_extra"],
        })

        result = split_datasets(
            [rgb_root, depth_root, normals_root],
            ["train", "val"],
            [70, 30],
            seed=99,
        )

        # For each suffix, collect qualified IDs from each modality's
        # written output.json and verify they match.
        for split_idx in range(2):
            suffix = ["train", "val"][split_idx]
            qid_sets: list[set[tuple[str, ...]]] = []
            for src_name in ["rgb", "depth", "normals"]:
                target = tmp_path / f"{src_name}_{suffix}"
                with open(target / ".ds_crawler" / "output.json") as f:
                    idx = json.load(f)
                qid_sets.append(collect_qualified_ids(idx))

            # All modalities have identical qualified IDs for this split
            assert qid_sets[0] == qid_sets[1] == qid_sets[2], (
                f"Qualified IDs mismatch across modalities for {suffix}"
            )

    def test_files_on_disk_match_output_json_all_targets(
        self, tmp_path: Path,
    ) -> None:
        """Every file in every split's output.json exists on disk."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {
            "scene:A": ["001", "002", "003"],
            "scene:B": ["001"],
        })
        self._build_source(depth_root, {
            "scene:A": ["001", "002"],
            "scene:B": ["001", "002"],
        })

        split_datasets(
            [rgb_root, depth_root], ["train", "val"], [70, 30],
        )

        for name in ["rgb_train", "rgb_val", "depth_train", "depth_val"]:
            target = tmp_path / name
            with open(target / ".ds_crawler" / "output.json") as f:
                idx = json.load(f)
            for rel_path in get_files(idx):
                assert (target / rel_path).is_file(), (
                    f"{rel_path} missing in {name}"
                )

    # -- per-source result metadata --

    def test_per_source_reports_exclusion_counts(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {
            "scene:A": ["001", "002", "003"],  # 003 is rgb-only
            "scene:B": ["001"],
        })
        self._build_source(depth_root, {
            "scene:A": ["001", "002"],
            "scene:B": ["001", "002"],          # 002 is depth-only
        })

        result = split_datasets(
            [rgb_root, depth_root], ["train", "val"], [50, 50],
        )

        rgb_info = result["per_source"][0]
        assert rgb_info["total_ids"] == 4
        assert rgb_info["excluded_ids"] == 1  # scene:A/003

        depth_info = result["per_source"][1]
        assert depth_info["total_ids"] == 4
        assert depth_info["excluded_ids"] == 1  # scene:B/002

    def test_per_source_split_metadata(self, tmp_path: Path) -> None:
        src = tmp_path / "only"
        self._build_source(src, {"s:A": ["1", "2"]})

        result = split_datasets([src], ["train", "val"], [50, 50])

        splits = result["per_source"][0]["splits"]
        assert len(splits) == 2
        assert splits[0]["suffix"] == "train"
        assert splits[0]["ratio"] == 50
        assert splits[1]["suffix"] == "val"
        assert splits[1]["ratio"] == 50
        for s in splits:
            assert s["missing"] == 0

    # -- three-way split --

    def test_three_way_split(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        scenes = {"s:A": [str(i) for i in range(10)]}
        self._build_source(rgb_root, scenes)
        self._build_source(depth_root, scenes)

        result = split_datasets(
            [rgb_root, depth_root],
            ["train", "val", "test"],
            [70, 20, 10],
            seed=1,
        )

        assert len(result["qualified_id_splits"]) == 3
        all_ids: set[tuple[str, ...]] = set()
        for s in result["qualified_id_splits"]:
            assert all_ids & s == set()
            all_ids |= s
        assert all_ids == result["common_ids"]

        # Each source has 3 splits
        for ps in result["per_source"]:
            assert len(ps["splits"]) == 3

    # -- seed determinism --

    def test_seed_deterministic(self, tmp_path: Path) -> None:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        scenes = {"s:A": ["1", "2", "3", "4"]}
        self._build_source(rgb_root, scenes)
        self._build_source(depth_root, scenes)

        r1 = split_datasets(
            [rgb_root, depth_root], ["train", "val"], [60, 40], seed=42,
        )
        # Re-run: targets already exist but output.json will be overwritten
        r2 = split_datasets(
            [rgb_root, depth_root], ["train", "val"], [60, 40], seed=42,
        )
        assert r1["qualified_id_splits"] == r2["qualified_id_splits"]

    # -- validation errors --

    def test_raises_on_mismatched_suffixes_ratios(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="same length"):
            split_datasets([tmp_path], ["train", "val"], [100])

    def test_raises_on_empty_sources(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            split_datasets([], ["train"], [100])

    def test_raises_on_missing_output_json(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="output.json"):
            split_datasets([root], ["train", "val"], [50, 50])

    # -- single source degenerates to split_dataset --

    def test_single_source(self, tmp_path: Path) -> None:
        """With one source, split_datasets behaves like split_dataset."""
        root = tmp_path / "only"
        idx = self._build_source(root, {"s:A": ["1", "2", "3", "4"]})

        result = split_datasets([root], ["train", "val"], [50, 50])

        all_qids = collect_qualified_ids(idx)
        assert result["common_ids"] == all_qids
        assert result["per_source"][0]["excluded_ids"] == 0

    # -- logging of excluded IDs --

    def test_exclusion_logged(self, tmp_path: Path, caplog) -> None:
        """IDs absent from the intersection are logged as warnings."""
        import logging

        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        self._build_source(rgb_root, {"s:A": ["1", "2", "extra"]})
        self._build_source(depth_root, {"s:A": ["1", "2"]})

        with caplog.at_level(logging.WARNING):
            split_datasets(
                [rgb_root, depth_root], ["train", "val"], [50, 50],
            )

        assert any(
            "not present in all sources" in rec.message
            for rec in caplog.records
        )
