"""Shared fixtures for dataset crawler tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.config import Config, DatasetConfig, _MODALITY_META_SCHEMAS

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_CONFIG_PATH = EXAMPLES_DIR / "config.json"
EXAMPLE_OUTPUT_PATH = EXAMPLES_DIR / "example_output.json"

# Default meta values used by test helpers when auto-injecting meta for a
# modality that requires it.
_DEFAULT_META: dict[str, dict[str, Any]] = {
    "depth": {"radial_depth": False, "scale_to_meters": 1.0},
    "rgb": {"range": [0, 255]},
    "semantic_segmentation": {"skyclass": [0, 0, 0]},
}


def default_meta_for_modality(modality: str) -> dict[str, Any] | None:
    """Return a valid default ``meta`` dict for *modality*, or ``None``."""
    return _DEFAULT_META.get(modality)


# ---------------------------------------------------------------------------
# Helpers for creating mock filesystem structures
# ---------------------------------------------------------------------------


def touch(path: Path) -> Path:
    """Create an empty file, including parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def create_vkitti2_tree(root: Path) -> list[Path]:
    """Create a mock VKITTI2 directory structure matching the example output.

    Returns list of all created files.
    """
    files = [
        # Scene01/clone - Camera_0 frames
        "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg",
        "Scene01/clone/frames/rgb/Camera_0/rgb_00002.jpg",
        # Scene01/clone - Camera_1 frames
        "Scene01/clone/frames/rgb/Camera_1/rgb_00001.jpg",
        # Scene01/clone - camera calibration
        "Scene01/clone/intrinsics/Camera_0_intrinsics.txt",
        "Scene01/clone/intrinsics/Camera_1_intrinsics.txt",
        "Scene01/clone/extrinsics/Camera_0_extrinsics.txt",
        "Scene01/clone/extrinsics/Camera_1_extrinsics.txt",
        # Scene02/fog - Camera_0 frames
        "Scene02/fog/frames/rgb/Camera_0/rgb_00001.jpg",
        # Scene02/fog - camera calibration
        "Scene02/fog/intrinsics/Camera_0_intrinsics.txt",
        "Scene02/fog/extrinsics/Camera_0_extrinsics.txt",
    ]
    return [touch(root / f) for f in files]


def create_ddad_tree(root: Path) -> list[Path]:
    """Create a mock DDAD directory structure matching the example output."""
    files = [
        "000150/rgb/CAMERA_01/0000000050.png",
        "000150/rgb/CAMERA_01/0000000051.png",
        "000150/rgb/CAMERA_05/0000000050.png",
        "000150/calibration/CAMERA_01.json",
        "000150/calibration/CAMERA_05.json",
    ]
    return [touch(root / f) for f in files]


def create_depth_predictions_tree(root: Path) -> list[Path]:
    """Create a mock depth_predictions directory structure matching the example output."""
    files = [
        "Scene01/00001_pred.png",
        "Scene01/00001_pred.npy",
        "Scene01/00002_pred.png",
        "Scene01/00002_pred.npy",
    ]
    return [touch(root / f) for f in files]


# ---------------------------------------------------------------------------
# Config builder helpers
# ---------------------------------------------------------------------------


def make_vkitti2_config(path: str) -> dict[str, Any]:
    """Return the VKITTI2 dataset config dict (matching examples/config.json).

    Adds .txt to file_extensions so the generic handler picks up camera files.
    """
    return {
        "name": "VKITTI2",
        "path": path,
        "type": "rgb",
        "file_extensions": [".jpg", ".txt"],
        "basename_regex": r"^rgb_(?P<frame>\d+)\.(?P<ext>jpg|png)$",
        "id_regex": (
            r"^(?P<scene>Scene\d+)/(?P<variation>[^/]+)/frames/rgb/"
            r"(?P<camera>Camera_\d+)/rgb_(?P<frame>\d+)\.(?:jpg|png)$"
        ),
        "id_regex_join_char": "+",
        "path_regex": (
            r"^(?P<scene>Scene\d+)/(?P<variation>[^/]+)/frames/rgb/"
            r"(?P<camera>Camera_\d+)/"
        ),
        "hierarchy_regex": (
            r"^(?P<scene>Scene\d+)/(?P<variation>[^/]+)/frames/rgb/"
            r"(?P<camera>Camera_\d+)/rgb_(?P<frame>\d+)\.(?:jpg|png)$"
        ),
        "named_capture_group_value_separator": ":",
        "intrinsics_regex": (
            r"^(?P<scene>Scene\d+)/(?P<variation>[^/]+)/intrinsics/"
            r"(?P<camera>Camera_\d+)_intrinsics\.txt$"
        ),
        "extrinsics_regex": (
            r"^(?P<scene>Scene\d+)/(?P<variation>[^/]+)/extrinsics/"
            r"(?P<camera>Camera_\d+)_extrinsics\.txt$"
        ),
        "flat_ids_unique": True,
        "properties": {
            "gt": True,
            "baseline": True,
            "euler_train": {
                "used_as": "target",
                "slot": "demo.target.rgb",
                "modality_type": "rgb",
            },
            "meta": {
                "range": [0, 255],
            },
            "dataset": {
                "license": "CC BY-NC-SA 4.0",
                "source": "https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
            },
        },
    }


def make_ddad_config(path: str) -> dict[str, Any]:
    """Return the DDAD dataset config dict.

    Adds .json to file_extensions so camera calibration files are found.
    """
    return {
        "name": "DDAD",
        "path": path,
        "type": "rgb",
        "file_extensions": [".png", ".json"],
        "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
        "id_regex": r"^(?P<scene>\d+)/rgb/(?P<camera>CAMERA_\d+)/(?P<frame>\d+)\.png$",
        "path_regex": r"^(?P<scene>\d+)/rgb/(?P<camera>CAMERA_\d+)/",
        "hierarchy_regex": r"^(?P<scene>\d+)/rgb/(?P<camera>CAMERA_\d+)/(?P<frame>\d+)\.png$",
        "named_capture_group_value_separator": ":",
        "intrinsics_regex": r"^(?P<scene>\d+)/calibration/(?P<camera>CAMERA_\d+)\.json$",
        "properties": {
            "gt": True,
            "euler_train": {
                "used_as": "target",
                "slot": "demo.target.rgb",
                "modality_type": "rgb",
            },
            "meta": {
                "range": [0, 255],
            },
        },
    }


def make_depth_predictions_config(path: str) -> dict[str, Any]:
    """Return the depth_predictions dataset config dict."""
    return {
        "name": "depth_predictions",
        "path": path,
        "type": "depth",
        "file_extensions": [".png", ".npy"],
        "basename_regex": r"^(?P<frame>\d+)_pred\.(?P<ext>png|npy)$",
        "id_regex": r"^(?P<scene>[^/]+)/(?P<frame>\d+)_pred\.(?P<ext>png|npy)$",
        "hierarchy_regex": r"^(?P<scene>[^/]+)/(?P<frame>\d+)_pred\.(?:png|npy)$",
        "named_capture_group_value_separator": ":",
        "output_json": None,
        "properties": {
            "gt": False,
            "model": "DepthAnythingV2",
            "euler_train": {
                "used_as": "input",
                "slot": "demo.input.depth",
                "modality_type": "depth",
            },
            "meta": {
                "radial_depth": False,
                "scale_to_meters": 1.0,
            },
        },
    }


def write_config_json(path: Path, datasets: list[dict[str, Any]]) -> Path:
    """Write a config JSON file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"datasets": datasets}, f)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def example_config_path() -> Path:
    """Path to the shipped example config.json."""
    return EXAMPLE_CONFIG_PATH


@pytest.fixture
def example_output_path() -> Path:
    """Path to the shipped example_output.json."""
    return EXAMPLE_OUTPUT_PATH


@pytest.fixture
def example_output_data(example_output_path: Path) -> list[dict]:
    """Parsed example output JSON."""
    with open(example_output_path) as f:
        return json.load(f)


@pytest.fixture
def vkitti2_root(tmp_path: Path) -> Path:
    """Temporary VKITTI2 dataset directory with mock files."""
    root = tmp_path / "vkitti2"
    create_vkitti2_tree(root)
    return root


@pytest.fixture
def ddad_root(tmp_path: Path) -> Path:
    """Temporary DDAD dataset directory with mock files."""
    root = tmp_path / "ddad"
    create_ddad_tree(root)
    return root


@pytest.fixture
def depth_predictions_root(tmp_path: Path) -> Path:
    """Temporary depth_predictions dataset directory with mock files."""
    root = tmp_path / "depth_predictions"
    create_depth_predictions_tree(root)
    return root


@pytest.fixture
def vkitti2_dataset_config(vkitti2_root: Path) -> DatasetConfig:
    """DatasetConfig for the mock VKITTI2 dataset."""
    cfg = make_vkitti2_config(str(vkitti2_root))
    return DatasetConfig(**cfg)


@pytest.fixture
def ddad_dataset_config(ddad_root: Path) -> DatasetConfig:
    """DatasetConfig for the mock DDAD dataset."""
    cfg = make_ddad_config(str(ddad_root))
    return DatasetConfig(**cfg)


@pytest.fixture
def depth_predictions_dataset_config(depth_predictions_root: Path) -> DatasetConfig:
    """DatasetConfig for the mock depth_predictions dataset."""
    cfg = make_depth_predictions_config(str(depth_predictions_root))
    return DatasetConfig(**cfg)


@pytest.fixture
def full_config(
    vkitti2_root: Path, ddad_root: Path, depth_predictions_root: Path, tmp_path: Path
) -> Config:
    """Full Config with all three datasets, backed by mock filesystem."""
    datasets_data = [
        make_vkitti2_config(str(vkitti2_root)),
        make_ddad_config(str(ddad_root)),
        make_depth_predictions_config(str(depth_predictions_root)),
    ]
    config_path = write_config_json(tmp_path / "config.json", datasets_data)
    return Config.from_file(config_path)


# ---------------------------------------------------------------------------
# ZIP helpers and fixtures
# ---------------------------------------------------------------------------


def create_zip_from_tree(
    root: Path, zip_path: Path, *, root_prefix: str = ""
) -> Path:
    """Create a ZIP archive containing every file under *root*.

    Directory entries are omitted; only files are stored with paths
    relative to *root*.

    When *root_prefix* is set (e.g. ``"mydata/"``), every entry is
    stored under that prefix — mimicking what macOS Compress does when
    you zip a folder.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, root_prefix + str(file.relative_to(root)))
    return zip_path


def create_zip_from_tree_with_config(
    root: Path, zip_path: Path, config: dict[str, Any], *, root_prefix: str = ""
) -> Path:
    """Create a ZIP with dataset files *and* a ``ds-crawler.json`` inside."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, root_prefix + str(file.relative_to(root)))
        zf.writestr(root_prefix + "ds-crawler.json", json.dumps(config))
    return zip_path


@pytest.fixture
def vkitti2_zip(tmp_path: Path) -> Path:
    """ZIP archive containing the mock VKITTI2 dataset."""
    root = tmp_path / "_vkitti2_tree"
    create_vkitti2_tree(root)
    return create_zip_from_tree(root, tmp_path / "vkitti2.zip")


@pytest.fixture
def depth_predictions_zip(tmp_path: Path) -> Path:
    """ZIP archive containing the mock depth_predictions dataset."""
    root = tmp_path / "_dp_tree"
    create_depth_predictions_tree(root)
    return create_zip_from_tree(root, tmp_path / "depth_predictions.zip")


@pytest.fixture
def vkitti2_prefixed_zip(tmp_path: Path) -> Path:
    """ZIP archive with a root directory prefix (macOS-style).

    The zip is named ``vkitti2.zip`` and entries live under ``vkitti2/``,
    matching how macOS Compress creates archives.
    """
    root = tmp_path / "_vkitti2_tree"
    create_vkitti2_tree(root)
    return create_zip_from_tree(
        root, tmp_path / "vkitti2.zip", root_prefix="vkitti2/"
    )


@pytest.fixture
def depth_predictions_prefixed_zip(tmp_path: Path) -> Path:
    """ZIP archive with a root directory prefix (macOS-style).

    The zip is named ``depth_predictions.zip`` and entries live under
    ``depth_predictions/``.
    """
    root = tmp_path / "_dp_tree"
    create_depth_predictions_tree(root)
    return create_zip_from_tree(
        root, tmp_path / "depth_predictions.zip", root_prefix="depth_predictions/"
    )
