"""Tests for copy_dataset and _collect_all_referenced_paths."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.parser import copy_dataset, get_files, index_dataset

from .conftest import (
    create_vkitti2_tree,
    make_vkitti2_config,
    make_depth_predictions_config,
    create_depth_predictions_tree,
    create_zip_from_tree,
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

    def test_sample_parameter(self, tmp_path: Path) -> None:
        """copy_dataset(sample=N) copies every Nth data file."""
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        all_files = get_files(idx)
        assert len(all_files) >= 4, "need enough files to test sampling"

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx, sample=2)

        # Every 2nd file from sorted list → roughly half
        expected_count = len(sorted(set(all_files))[::2])
        assert summary["copied"] == expected_count
        assert summary["missing"] == 0

    def test_sample_preserves_camera_files(self, tmp_path: Path) -> None:
        """sample on copy_dataset still copies camera intrinsics/extrinsics."""
        root = tmp_path / "src"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(root, dst, index=idx, sample=100)

        # With sample=100 most data files are dropped, but camera files remain
        assert summary["missing"] == 0
        assert (dst / "Scene01/clone/intrinsics/Camera_0_intrinsics.txt").is_file()
        assert (dst / "Scene01/clone/extrinsics/Camera_0_extrinsics.txt").is_file()

    def test_sample_one_copies_all(self, tmp_path: Path) -> None:
        """sample=1 should copy every file (no subsampling)."""
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst_all = tmp_path / "dst_all"
        summary_all = copy_dataset(root, dst_all, index=idx)

        dst_s1 = tmp_path / "dst_s1"
        summary_s1 = copy_dataset(root, dst_s1, index=idx, sample=1)

        assert summary_s1["copied"] == summary_all["copied"]

    def test_sample_filters_output_json(self, tmp_path: Path) -> None:
        """output.json written by copy_dataset(sample=N) only has sampled files."""
        root = tmp_path / "src"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        all_files = get_files(idx)
        assert len(all_files) >= 4

        dst = tmp_path / "dst"
        copy_dataset(root, dst, index=idx, sample=2)

        with open(dst / "output.json") as f:
            written_idx = json.load(f)

        written_files = get_files(written_idx)
        expected_count = len(sorted(set(all_files))[::2])
        assert len(written_files) == expected_count
        # Every written file path must actually exist on disk
        for rel_path in written_files:
            assert (dst / rel_path).is_file()

    def test_sample_filters_output_json_with_cameras(self, tmp_path: Path) -> None:
        """Sampled output.json preserves camera paths but filters data files."""
        root = tmp_path / "src"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        idx = index_dataset(cfg)

        full_files = get_files(idx)

        dst = tmp_path / "dst"
        copy_dataset(root, dst, index=idx, sample=100)

        with open(dst / "output.json") as f:
            written_idx = json.load(f)

        written_files = get_files(written_idx)
        # With sample=100, most data files are dropped
        assert len(written_files) < len(full_files)
        # But camera info is still present in the hierarchy
        # (hierarchy keys use "key:value" naming)
        dataset = written_idx.get("dataset", {})
        scene = dataset["children"]["scene:Scene01"]
        variation = scene["children"]["variation:clone"]
        camera = variation["children"]["camera:Camera_0"]
        assert "camera_intrinsics" in camera
        assert "camera_extrinsics" in camera

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


class TestCopyDatasetFromZip:
    """Tests for copy_dataset when the input is a .zip archive."""

    @staticmethod
    def _make_zip_with_output_json(
        tmp_path: Path, *, root_prefix: str = ""
    ) -> tuple[Path, dict[str, Any]]:
        """Build a zip containing dataset files and an embedded output.json."""
        root = tmp_path / "_tree"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        zip_path = tmp_path / "depth_predictions.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    zf.write(file, root_prefix + str(file.relative_to(root)))
            zf.writestr(
                root_prefix + "output.json", json.dumps(idx, indent=2)
            )
        return zip_path, idx

    def test_copies_all_files_from_zip(self, tmp_path: Path) -> None:
        """copy_dataset can copy files out of a zip with an explicit index."""
        root = tmp_path / "_tree"
        create_depth_predictions_tree(root)
        zip_path = create_zip_from_tree(root, tmp_path / "ds.zip")
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(zip_path, dst, index=idx)

        expected_files = get_files(idx)
        assert summary["copied"] == len(expected_files)
        assert summary["missing"] == 0
        for rel_path in expected_files:
            assert (dst / rel_path).is_file()

    def test_loads_output_json_from_zip(self, tmp_path: Path) -> None:
        """copy_dataset(index=None) reads output.json from inside the zip."""
        zip_path, idx = self._make_zip_with_output_json(tmp_path)

        dst = tmp_path / "dst"
        summary = copy_dataset(zip_path, dst)

        assert summary["copied"] > 0
        assert summary["missing"] == 0

    def test_loads_output_json_from_prefixed_zip(self, tmp_path: Path) -> None:
        """output.json is found even when the zip has a root prefix."""
        zip_path, idx = self._make_zip_with_output_json(
            tmp_path, root_prefix="depth_predictions/"
        )

        dst = tmp_path / "dst"
        summary = copy_dataset(zip_path, dst)

        assert summary["copied"] > 0
        assert summary["missing"] == 0

    def test_raises_when_no_index_in_zip(self, tmp_path: Path) -> None:
        """FileNotFoundError when zip has no output.json and index=None."""
        root = tmp_path / "_tree"
        create_depth_predictions_tree(root)
        zip_path = create_zip_from_tree(root, tmp_path / "ds.zip")

        dst = tmp_path / "dst"
        with pytest.raises(FileNotFoundError, match="No output.json found"):
            copy_dataset(zip_path, dst)

    def test_writes_index_to_output_dir(self, tmp_path: Path) -> None:
        zip_path, idx = self._make_zip_with_output_json(tmp_path)

        dst = tmp_path / "dst"
        copy_dataset(zip_path, dst)

        assert (dst / "output.json").is_file()
        with open(dst / "output.json") as f:
            written = json.load(f)
        assert written == idx

    def test_copies_from_prefixed_zip_with_explicit_index(
        self, tmp_path: Path
    ) -> None:
        """Files are extracted correctly from a zip with a root prefix."""
        root = tmp_path / "_tree"
        create_depth_predictions_tree(root)
        zip_path = create_zip_from_tree(
            root,
            tmp_path / "depth_predictions.zip",
            root_prefix="depth_predictions/",
        )
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        dst = tmp_path / "dst"
        summary = copy_dataset(zip_path, dst, index=idx)

        expected_files = get_files(idx)
        assert summary["copied"] == len(expected_files)
        assert summary["missing"] == 0
        for rel_path in expected_files:
            assert (dst / rel_path).is_file()

    def test_sample_parameter_with_zip(self, tmp_path: Path) -> None:
        """copy_dataset(sample=N) works with zip input."""
        root = tmp_path / "_tree"
        create_depth_predictions_tree(root)
        zip_path = create_zip_from_tree(root, tmp_path / "ds.zip")
        cfg = make_depth_predictions_config(str(root))
        idx = index_dataset(cfg)

        all_files = get_files(idx)
        assert len(all_files) >= 4

        dst = tmp_path / "dst"
        summary = copy_dataset(zip_path, dst, index=idx, sample=2)

        expected_count = len(sorted(set(all_files))[::2])
        assert summary["copied"] == expected_count
        assert summary["missing"] == 0
