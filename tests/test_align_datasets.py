"""Tests for align_datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler import align_datasets, index_dataset
from .conftest import touch


# ---------------------------------------------------------------------------
# Helpers – tiny datasets with overlapping IDs
# ---------------------------------------------------------------------------

_SHARED_ID_REGEX = r"^(?P<scene>[^/]+)/(?P<frame>\d+)\.\w+$"
_SHARED_HIERARCHY_REGEX = r"^(?P<scene>[^/]+)/(?P<frame>\d+)\.\w+$"
_SEPARATOR = ":"


def _make_rgb_tree(root: Path) -> list[Path]:
    files = [
        "scene01/001.jpg",
        "scene01/002.jpg",
        "scene02/001.jpg",
    ]
    return [touch(root / f) for f in files]


def _make_depth_tree(root: Path) -> list[Path]:
    files = [
        "scene01/001.exr",
        "scene01/002.exr",
        # scene02/003 exists only in depth (no rgb counterpart)
        "scene02/003.exr",
    ]
    return [touch(root / f) for f in files]


def _make_seg_tree(root: Path) -> list[Path]:
    """A third modality that partially overlaps."""
    files = [
        "scene01/001.png",
    ]
    return [touch(root / f) for f in files]


def _rgb_config(path: str) -> dict[str, Any]:
    return {
        "name": "rgb",
        "path": path,
        "type": "rgb",
        "file_extensions": [".jpg"],
        "id_regex": _SHARED_ID_REGEX,
        "hierarchy_regex": _SHARED_HIERARCHY_REGEX,
        "named_capture_group_value_separator": _SEPARATOR,
        "properties": {
            "euler_train": {
                "used_as": "input",
                "modality_type": "rgb",
            }
        },
    }


def _depth_config(path: str) -> dict[str, Any]:
    return {
        "name": "depth",
        "path": path,
        "type": "depth",
        "file_extensions": [".exr"],
        "id_regex": _SHARED_ID_REGEX,
        "hierarchy_regex": _SHARED_HIERARCHY_REGEX,
        "named_capture_group_value_separator": _SEPARATOR,
        "properties": {
            "euler_train": {
                "used_as": "target",
                "modality_type": "depth",
            },
            "meta": {
                "radial_depth": False,
                "scale_to_meters": 1.0,
            },
        },
    }


def _seg_config(path: str) -> dict[str, Any]:
    return {
        "name": "seg",
        "path": path,
        "type": "segmentation",
        "file_extensions": [".png"],
        "id_regex": _SHARED_ID_REGEX,
        "hierarchy_regex": _SHARED_HIERARCHY_REGEX,
        "named_capture_group_value_separator": _SEPARATOR,
        "properties": {
            "euler_train": {
                "used_as": "target",
                "modality_type": "segmentation",
            }
        },
    }


def _index(config: dict[str, Any]) -> dict[str, Any]:
    """Convenience: index a dataset and return the output dict."""
    return index_dataset(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlignDatasetsWithDicts:
    """align_datasets with pre-loaded output JSON dicts."""

    def test_full_overlap(self, tmp_path: Path) -> None:
        """Two modalities with identical IDs produce complete pairs."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        # Create identical file sets (different extensions)
        for name in ("scene01/001", "scene01/002"):
            touch(rgb_root / f"{name}.jpg")
            touch(depth_root / f"{name}.exr")

        rgb_out = _index(_rgb_config(str(rgb_root)))
        depth_out = _index(_depth_config(str(depth_root)))

        aligned = align_datasets(
            {"modality": "rgb", "source": rgb_out},
            {"modality": "depth", "source": depth_out},
        )

        assert len(aligned) == 2
        for file_id, modalities in aligned.items():
            assert "rgb" in modalities
            assert "depth" in modalities
            assert modalities["rgb"]["id"] == modalities["depth"]["id"]

    def test_partial_overlap(self, tmp_path: Path) -> None:
        """IDs unique to one modality still appear, with only that modality."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        _make_rgb_tree(rgb_root)
        _make_depth_tree(depth_root)

        rgb_out = _index(_rgb_config(str(rgb_root)))
        depth_out = _index(_depth_config(str(depth_root)))

        aligned = align_datasets(
            {"modality": "rgb", "source": rgb_out},
            {"modality": "depth", "source": depth_out},
        )

        # rgb has: scene01/001, scene01/002, scene02/001
        # depth has: scene01/001, scene01/002, scene02/003
        # union = 4 unique IDs
        assert len(aligned) == 4

        # Count entries present in both modalities
        both = [v for v in aligned.values() if len(v) == 2]
        assert len(both) == 2  # scene01/001 and scene01/002

        # scene02/001 only in rgb
        rgb_only = [v for v in aligned.values() if list(v.keys()) == ["rgb"]]
        assert len(rgb_only) == 1

        # scene02/003 only in depth
        depth_only = [v for v in aligned.values() if list(v.keys()) == ["depth"]]
        assert len(depth_only) == 1

    def test_three_modalities(self, tmp_path: Path) -> None:
        """Alignment works with more than two modalities."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        seg_root = tmp_path / "seg"
        _make_rgb_tree(rgb_root)
        _make_depth_tree(depth_root)
        _make_seg_tree(seg_root)

        rgb_out = _index(_rgb_config(str(rgb_root)))
        depth_out = _index(_depth_config(str(depth_root)))
        seg_out = _index(_seg_config(str(seg_root)))

        aligned = align_datasets(
            {"modality": "rgb", "source": rgb_out},
            {"modality": "depth", "source": depth_out},
            {"modality": "seg", "source": seg_out},
        )

        # scene01/001 is the only ID in all three
        all_three = [v for v in aligned.values() if len(v) == 3]
        assert len(all_three) == 1

    def test_empty_args(self) -> None:
        """No arguments returns empty dict."""
        assert align_datasets() == {}

    def test_single_modality(self, tmp_path: Path) -> None:
        """A single modality returns all its entries keyed by ID."""
        rgb_root = tmp_path / "rgb"
        _make_rgb_tree(rgb_root)
        rgb_out = _index(_rgb_config(str(rgb_root)))

        aligned = align_datasets({"modality": "rgb", "source": rgb_out})

        assert len(aligned) == 3
        for modalities in aligned.values():
            assert list(modalities.keys()) == ["rgb"]

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        """Result keys are sorted by file ID."""
        rgb_root = tmp_path / "rgb"
        _make_rgb_tree(rgb_root)
        rgb_out = _index(_rgb_config(str(rgb_root)))

        aligned = align_datasets({"modality": "rgb", "source": rgb_out})
        ids = list(aligned.keys())
        assert ids == sorted(ids)


class TestAlignDatasetsFromPath:
    """align_datasets resolving sources from filesystem paths."""

    def test_resolve_from_output_json(self, tmp_path: Path) -> None:
        """Source path with an existing output.json is loaded directly."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        _make_rgb_tree(rgb_root)
        _make_depth_tree(depth_root)

        # Index and save output.json
        rgb_out = index_dataset(_rgb_config(str(rgb_root)), save_index=True)
        depth_out = index_dataset(_depth_config(str(depth_root)), save_index=True)

        # Now align using paths — should pick up the saved output.json
        aligned = align_datasets(
            {"modality": "rgb", "source": str(rgb_root)},
            {"modality": "depth", "source": str(depth_root)},
        )

        assert len(aligned) == 4  # same as partial overlap test

    def test_resolve_from_ds_crawler_json(self, tmp_path: Path) -> None:
        """Source path with ds-crawler.json (no output.json) indexes on the fly."""
        rgb_root = tmp_path / "rgb"
        _make_rgb_tree(rgb_root)

        # Write ds-crawler.json instead of output.json
        config = _rgb_config(str(rgb_root))
        with open(rgb_root / "ds-crawler.json", "w") as f:
            json.dump(config, f)

        aligned = align_datasets({"modality": "rgb", "source": str(rgb_root)})

        assert len(aligned) == 3

    def test_resolve_missing_raises(self, tmp_path: Path) -> None:
        """Path with neither output.json nor ds-crawler.json raises."""
        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        with pytest.raises(FileNotFoundError):
            align_datasets({"modality": "rgb", "source": str(empty_root)})

    def test_mixed_dict_and_path(self, tmp_path: Path) -> None:
        """One source as dict, another as path."""
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        _make_rgb_tree(rgb_root)
        _make_depth_tree(depth_root)

        rgb_out = _index(_rgb_config(str(rgb_root)))
        index_dataset(_depth_config(str(depth_root)), save_index=True)

        aligned = align_datasets(
            {"modality": "rgb", "source": rgb_out},
            {"modality": "depth", "source": str(depth_root)},
        )

        assert len(aligned) == 4
        both = [v for v in aligned.values() if len(v) == 2]
        assert len(both) == 2
