"""Tests for extract_datasets."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ds_crawler import extract_datasets, collect_qualified_ids, get_files
from ds_crawler.zip_utils import METADATA_DIR, OUTPUT_FILENAME

from .conftest import touch


# ---------------------------------------------------------------------------
# Helpers: mixed-source dataset with RGB and depth side by side
# ---------------------------------------------------------------------------


def _create_mixed_tree(root: Path) -> list[Path]:
    """Create a directory with both RGB and depth files in the same folders.

    Structure::

        scene01/rgb_001.png
        scene01/rgb_002.png
        scene01/depth_001.png
        scene01/depth_002.png
        scene01/depth_003.png      <-- no RGB counterpart
        scene02/rgb_001.png
        scene02/depth_001.png
    """
    files = [
        "scene01/rgb_001.png",
        "scene01/rgb_002.png",
        "scene01/depth_001.png",
        "scene01/depth_002.png",
        "scene01/depth_003.png",
        "scene02/rgb_001.png",
        "scene02/depth_001.png",
    ]
    return [touch(root / f) for f in files]


def _make_rgb_config(path: str) -> dict[str, Any]:
    return {
        "name": "rgb",
        "path": path,
        "type": "rgb",
        "file_extensions": [".png"],
        "id_regex": r"^(?P<scene>[^/]+)/rgb_(?P<frame>\d+)\.png$",
        "hierarchy_regex": r"^(?P<scene>[^/]+)/rgb_(?P<frame>\d+)\.png$",
        "named_capture_group_value_separator": ":",
        "properties": {
            "euler_train": {
                "used_as": "target",
                "slot": "test.target.rgb",
                "modality_type": "rgb",
            },
            "meta": {"range": [0, 255]},
        },
    }


def _make_depth_config(path: str) -> dict[str, Any]:
    return {
        "name": "depth",
        "path": path,
        "type": "depth",
        "file_extensions": [".png"],
        "id_regex": r"^(?P<scene>[^/]+)/depth_(?P<frame>\d+)\.png$",
        "hierarchy_regex": r"^(?P<scene>[^/]+)/depth_(?P<frame>\d+)\.png$",
        "named_capture_group_value_separator": ":",
        "properties": {
            "euler_train": {
                "used_as": "target",
                "slot": "test.target.depth",
                "modality_type": "depth",
            },
            "meta": {
                "radial_depth": False,
                "scale_to_meters": 1.0,
                "range": [0, 65535],
            },
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestExtractDatasetsValidation:
    """Input validation."""

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            extract_datasets([{"name": "a"}], ["/out1", "/out2"])

    def test_raises_on_empty_configs(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extract_datasets([], [])


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestExtractDatasetsBasic:
    """Core extraction behaviour."""

    def test_single_config_extracts_matched_files(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        out = tmp_path / "rgb_out"

        result = extract_datasets([rgb_cfg], [out])

        assert len(result["extractions"]) == 1
        ext = result["extractions"][0]
        # RGB regex matches 3 files (scene01/rgb_001, scene01/rgb_002, scene02/rgb_001)
        assert ext["copied"] == 3
        assert ext["missing"] == 0
        assert ext["config_name"] == "rgb"

    def test_two_configs_from_same_source(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        result = extract_datasets(
            [rgb_cfg, depth_cfg], [rgb_out, depth_out],
        )

        assert len(result["extractions"]) == 2
        assert result["extractions"][0]["copied"] == 3   # 3 RGB files
        assert result["extractions"][1]["copied"] == 4   # 4 depth files

    def test_output_json_written_in_each_target(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        extract_datasets([rgb_cfg, depth_cfg], [rgb_out, depth_out])

        rgb_index_path = rgb_out / METADATA_DIR / OUTPUT_FILENAME
        depth_index_path = depth_out / METADATA_DIR / OUTPUT_FILENAME
        assert rgb_index_path.is_file()
        assert depth_index_path.is_file()

        with open(rgb_index_path) as f:
            rgb_index = json.load(f)
        assert rgb_index["name"] == "rgb"
        assert rgb_index["type"] == "rgb"

        with open(depth_index_path) as f:
            depth_index = json.load(f)
        assert depth_index["name"] == "depth"
        assert depth_index["type"] == "depth"

    def test_extracted_files_exist_on_disk(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        out = tmp_path / "rgb_out"

        extract_datasets([rgb_cfg], [out])

        index_path = out / METADATA_DIR / OUTPUT_FILENAME
        with open(index_path) as f:
            index = json.load(f)
        for rel_path in get_files(index):
            assert (out / rel_path).is_file(), f"Missing: {rel_path}"

    def test_zip_output(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        out_zip = tmp_path / "rgb_out.zip"

        result = extract_datasets([rgb_cfg], [out_zip])

        assert out_zip.is_file()
        assert result["extractions"][0]["copied"] == 3
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert f"{METADATA_DIR}/{OUTPUT_FILENAME}" in names
            # Data files should be in the zip
            assert len([n for n in names if n.endswith(".png")]) == 3


# ---------------------------------------------------------------------------
# Intersection warnings
# ---------------------------------------------------------------------------


class TestExtractDatasetsIntersection:
    """Intersection computation and warnings."""

    def test_complete_intersection_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When all configs match the same IDs, no warning is logged."""
        src = tmp_path / "source"
        # Create files where both modalities have the same set of IDs
        touch(src / "s01/rgb_001.png")
        touch(src / "s01/depth_001.png")

        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        with caplog.at_level(logging.WARNING):
            result = extract_datasets(
                [rgb_cfg, depth_cfg], [rgb_out, depth_out],
            )

        assert result["incomplete_ids"] == {}
        assert not any(
            "incomplete intersection" in rec.message for rec in caplog.records
        )

    def test_incomplete_intersection_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """IDs present in one config but not another trigger a warning."""
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        with caplog.at_level(logging.WARNING):
            extract_datasets(
                [rgb_cfg, depth_cfg], [rgb_out, depth_out],
            )

        assert any(
            "incomplete intersection" in rec.message for rec in caplog.records
        )

    def test_incomplete_ids_populated(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        result = extract_datasets(
            [rgb_cfg, depth_cfg], [rgb_out, depth_out],
        )

        # depth has scene01/depth_003 which has no RGB counterpart
        assert "depth" in result["incomplete_ids"]
        assert len(result["incomplete_ids"]["depth"]) == 1
        # RGB has no extras (all RGB IDs also exist in depth)
        assert "rgb" not in result["incomplete_ids"]

    def test_common_ids_is_intersection(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        result = extract_datasets(
            [rgb_cfg, depth_cfg], [rgb_out, depth_out],
        )

        rgb_ids = result["per_config_ids"][0]
        depth_ids = result["per_config_ids"][1]
        assert result["common_ids"] == rgb_ids & depth_ids


# ---------------------------------------------------------------------------
# Return value structure
# ---------------------------------------------------------------------------


class TestExtractDatasetsResult:
    """Return value contains expected keys and values."""

    def test_extraction_result_keys(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        out = tmp_path / "rgb_out"

        result = extract_datasets([rgb_cfg], [out])

        ext = result["extractions"][0]
        assert "config_name" in ext
        assert "source" in ext
        assert "target" in ext
        assert "num_ids" in ext
        assert "copied" in ext
        assert "missing" in ext
        assert "missing_files" in ext

    def test_per_config_ids_populated(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        depth_cfg = _make_depth_config(str(src))
        rgb_out = tmp_path / "rgb_out"
        depth_out = tmp_path / "depth_out"

        result = extract_datasets(
            [rgb_cfg, depth_cfg], [rgb_out, depth_out],
        )

        assert len(result["per_config_ids"]) == 2
        assert isinstance(result["per_config_ids"][0], set)
        assert isinstance(result["per_config_ids"][1], set)
        # RGB: 3 qualified IDs (scene01/001, scene01/002, scene02/001)
        assert len(result["per_config_ids"][0]) == 3
        # Depth: 4 qualified IDs
        assert len(result["per_config_ids"][1]) == 4

    def test_single_config_common_ids_equals_own_ids(self, tmp_path: Path) -> None:
        """With a single config the intersection is the config's own IDs."""
        src = tmp_path / "source"
        _create_mixed_tree(src)
        rgb_cfg = _make_rgb_config(str(src))
        out = tmp_path / "rgb_out"

        result = extract_datasets([rgb_cfg], [out])

        assert result["common_ids"] == result["per_config_ids"][0]
        assert result["incomplete_ids"] == {}
