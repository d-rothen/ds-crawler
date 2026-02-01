"""Tests for copy_dataset and _collect_all_referenced_paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.parser import copy_dataset, get_files, index_dataset

from .conftest import (
    create_vkitti2_tree,
    make_vkitti2_config,
    make_depth_predictions_config,
    create_depth_predictions_tree,
    touch,
)


class TestCopyDataset:
    def test_copies_all_files(self, tmp_path: Path) -> None:
        """All referenced files are copied to output_path."""
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx)

        expected_files = get_files(idx)
        assert summary["copied"] == len(expected_files)
        assert summary["missing"] == 0
        for rel_path in expected_files:
            assert (dst / rel_path).is_file()

    def test_preserves_directory_structure(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        copy_dataset(root, dst, index=idx)

        for rel_path in get_files(idx):
            assert (dst / rel_path).is_file()
            # Parent directory structure must exist
            assert (dst / rel_path).parent.is_dir()

    def test_copies_camera_files(self, tmp_path: Path) -> None:
        """camera_intrinsics and camera_extrinsics are also copied."""
        root = tmp_path / "src"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        copy_dataset(root, dst, index=idx)

        # Check that camera files were copied
        assert (dst / "Scene01/clone/intrinsics/Camera_0_intrinsics.txt").is_file()
        assert (dst / "Scene01/clone/extrinsics/Camera_0_extrinsics.txt").is_file()

    def test_loads_output_json_when_index_is_none(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        # Write output.json to source
        with open(root / "output.json", "w") as f:
            json.dump(idx, f)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst)  # index=None, loads from file

        assert summary["copied"] > 0
        assert summary["missing"] == 0

    def test_writes_index_to_output(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        copy_dataset(root, dst, index=idx)

        # output.json must exist in destination
        output_json_path = dst / "output.json"
        assert output_json_path.is_file()
        with open(output_json_path) as f:
            written_idx = json.load(f)
        assert written_idx == idx

    def test_missing_files_logged_and_skipped(self, tmp_path: Path) -> None:
        """Missing source files produce warnings, not errors."""
        root = tmp_path / "src"
        root.mkdir()
        # Create an index referencing files that don't exist
        idx: dict[str, Any] = {
            "name": "test",
            "dataset": {
                "files": [
                    {"path": "missing_a.png", "id": "a"},
                    {"path": "missing_b.png", "id": "b"},
                ]
            },
        }

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx)

        assert summary["copied"] == 0
        assert summary["missing"] == 2
        assert "missing_a.png" in summary["missing_files"]
        assert "missing_b.png" in summary["missing_files"]

    def test_returns_summary_dict(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx)

        assert "copied" in summary
        assert "missing" in summary
        assert "missing_files" in summary
        assert isinstance(summary["copied"], int)
        assert isinstance(summary["missing"], int)
        assert isinstance(summary["missing_files"], list)

    def test_raises_when_no_index_and_no_output_json(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "src"
        root.mkdir()
        dst = tmp_path / "dst"

        with pytest.raises(FileNotFoundError, match="No output.json found"):
            copy_dataset(root, dst)

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst" / "nested" / "deep"
        copy_dataset(root, dst, index=idx)

        assert dst.is_dir()
        assert (dst / "output.json").is_file()

    def test_with_vkitti2_dataset(self, tmp_path: Path) -> None:
        """End-to-end: index VKITTI2, copy it, verify structure."""
        root = tmp_path / "src"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx)

        # All data files + camera files should be copied
        assert summary["missing"] == 0
        assert summary["copied"] > 0
        # Verify a specific file
        assert (dst / "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg").is_file()

    def test_with_sampled_index(self, tmp_path: Path) -> None:
        """copy_dataset with an index that was produced with sample=N."""
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))

        full_idx = index_dataset(cfg)
        sampled_idx = index_dataset(cfg, sample=2)

        full_files = get_files(full_idx)
        sampled_files = get_files(sampled_idx)
        assert len(sampled_files) < len(full_files)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=sampled_idx)

        assert summary["copied"] == len(sampled_files)
        assert summary["missing"] == 0

    def test_deduplicates_camera_paths(self, tmp_path: Path) -> None:
        """Same camera file referenced at multiple levels is copied once."""
        root = tmp_path / "src"
        # Create a single camera file
        touch(root / "intrinsics.txt")

        # Index that references the same file at two nodes
        idx: dict[str, Any] = {
            "name": "test",
            "dataset": {
                "children": {
                    "a": {
                        "camera_intrinsics": "intrinsics.txt",
                        "files": [],
                    },
                    "b": {
                        "camera_intrinsics": "intrinsics.txt",
                        "files": [],
                    },
                }
            },
        }

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx)

        # Should be copied exactly once
        assert summary["copied"] == 1
        assert (dst / "intrinsics.txt").is_file()
