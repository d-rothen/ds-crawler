"""Tests for crawler.config – loading, validation, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_crawler.config import Config, DatasetConfig, DEFAULT_TYPE_EXTENSIONS


# ===================================================================
# Example config.json validation
# ===================================================================


class TestExampleConfigLoads:
    """Verify the shipped examples/config.json loads without errors."""

    def test_loads_successfully(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        assert len(config.datasets) == 3

    def test_dataset_names(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        names = [ds.name for ds in config.datasets]
        assert names == ["VKITTI2", "DDAD", "depth_predictions"]

    def test_dataset_types(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        types = [ds.type for ds in config.datasets]
        assert types == ["rgb", "rgb", "depth"]

    def test_vkitti2_regexes_compile(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        assert vk.compiled_basename_regex is not None
        assert vk.compiled_id_regex is not None
        assert vk.compiled_path_regex is not None
        assert vk.compiled_hierarchy_regex is not None
        assert vk.compiled_intrinsics_regex is not None
        assert vk.compiled_extrinsics_regex is not None

    def test_ddad_has_no_extrinsics(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        ddad = config.datasets[1]
        assert ddad.compiled_intrinsics_regex is not None
        assert ddad.compiled_extrinsics_regex is None

    def test_depth_predictions_has_no_camera_regexes(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        dp = config.datasets[2]
        assert dp.compiled_intrinsics_regex is None
        assert dp.compiled_extrinsics_regex is None

    def test_depth_predictions_custom_output_json(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        dp = config.datasets[2]
        assert dp.output_json == "/data/predictions/depth_index.json"

    def test_vkitti2_flat_ids_unique(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        assert vk.flat_ids_unique is True

    def test_ddad_flat_ids_unique_default(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        ddad = config.datasets[1]
        assert ddad.flat_ids_unique is False


# ===================================================================
# DatasetConfig construction and validation
# ===================================================================


class TestDatasetConfigValidation:
    """Test DatasetConfig field validation."""

    def _minimal_kwargs(self, **overrides) -> dict:
        """Return minimal valid kwargs for DatasetConfig."""
        defaults = {
            "name": "test",
            "path": "/tmp/test",
            "type": "rgb",
            "basename_regex": r"^(?P<name>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<name>.+)\.png$",
        }
        defaults.update(overrides)
        return defaults

    def test_valid_minimal_config(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs())
        assert ds.name == "test"
        assert ds.type == "rgb"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid type"):
            DatasetConfig(**self._minimal_kwargs(type="video"))

    def test_valid_types(self) -> None:
        for t in ("rgb", "depth", "segmentation"):
            ds = DatasetConfig(**self._minimal_kwargs(type=t))
            assert ds.type == t

    def test_invalid_basename_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid basename_regex"):
            DatasetConfig(**self._minimal_kwargs(basename_regex="[invalid"))

    def test_invalid_id_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid id_regex"):
            DatasetConfig(**self._minimal_kwargs(id_regex="[invalid"))

    def test_id_regex_no_capture_group_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one capture group"):
            DatasetConfig(**self._minimal_kwargs(id_regex=r"^no_group\.png$"))

    def test_invalid_path_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid path_regex"):
            DatasetConfig(**self._minimal_kwargs(path_regex="[invalid"))

    def test_invalid_hierarchy_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid hierarchy_regex"):
            DatasetConfig(**self._minimal_kwargs(hierarchy_regex="[invalid"))

    def test_hierarchy_regex_no_capture_group_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one capture group"):
            DatasetConfig(
                **self._minimal_kwargs(hierarchy_regex=r"^no_group\.png$")
            )

    def test_hierarchy_named_groups_without_separator_raises(self) -> None:
        with pytest.raises(
            ValueError, match="named_capture_group_value_separator"
        ):
            DatasetConfig(
                **self._minimal_kwargs(
                    hierarchy_regex=r"^(?P<name>.+)\.png$",
                    named_capture_group_value_separator=None,
                )
            )

    def test_hierarchy_named_groups_with_separator_ok(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                hierarchy_regex=r"^(?P<name>.+)\.png$",
                named_capture_group_value_separator=":",
            )
        )
        assert ds.compiled_hierarchy_regex is not None

    def test_hierarchy_unnamed_groups_without_separator_ok(self) -> None:
        """Unnamed capture groups don't require a separator."""
        ds = DatasetConfig(
            **self._minimal_kwargs(
                hierarchy_regex=r"^(.+)\.png$",
                named_capture_group_value_separator=None,
            )
        )
        assert ds.compiled_hierarchy_regex is not None

    def test_invalid_intrinsics_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid intrinsics_regex"):
            DatasetConfig(**self._minimal_kwargs(intrinsics_regex="[invalid"))

    def test_intrinsics_regex_no_capture_group_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one capture group"):
            DatasetConfig(
                **self._minimal_kwargs(intrinsics_regex=r"^intrinsics\.txt$")
            )

    def test_invalid_extrinsics_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid extrinsics_regex"):
            DatasetConfig(**self._minimal_kwargs(extrinsics_regex="[invalid"))

    def test_extrinsics_regex_no_capture_group_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one capture group"):
            DatasetConfig(
                **self._minimal_kwargs(extrinsics_regex=r"^extrinsics\.txt$")
            )


# ===================================================================
# File extension handling
# ===================================================================


class TestFileExtensions:
    """Test file extension normalization and defaults."""

    def _minimal_kwargs(self, **overrides) -> dict:
        defaults = {
            "name": "test",
            "path": "/tmp/test",
            "type": "rgb",
            "basename_regex": r"^(.+)\.(?P<ext>png)$",
            "id_regex": r"^(.+)\.png$",
        }
        defaults.update(overrides)
        return defaults

    def test_default_rgb_extensions(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="rgb"))
        assert ds.get_file_extensions() == {".png", ".jpg", ".jpeg"}

    def test_default_depth_extensions(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="depth"))
        assert ds.get_file_extensions() == {".png", ".exr", ".npy", ".pfm"}

    def test_default_segmentation_extensions(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="segmentation"))
        assert ds.get_file_extensions() == {".png"}

    def test_custom_extensions_override_defaults(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(file_extensions=[".tiff", ".bmp"])
        )
        assert ds.get_file_extensions() == {".tiff", ".bmp"}

    def test_extension_normalization_adds_dot(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(file_extensions=["png", "jpg"])
        )
        assert ds.file_extensions == [".png", ".jpg"]
        assert ds.get_file_extensions() == {".png", ".jpg"}

    def test_extension_normalization_preserves_dot(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(file_extensions=[".png", ".jpg"])
        )
        assert ds.file_extensions == [".png", ".jpg"]

    def test_empty_extensions_list(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(file_extensions=[]))
        assert ds.get_file_extensions() == set()


# ===================================================================
# Config.from_file
# ===================================================================


class TestConfigFromFile:
    """Test Config.from_file loading."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_file(tmp_path / "nonexistent.json")

    def test_missing_datasets_key_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        config_path.write_text('{"something": "else"}')
        with pytest.raises(ValueError, match="must contain 'datasets'"):
            Config.from_file(config_path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        data = {"datasets": [{"name": "test"}]}
        config_path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing required field"):
            Config.from_file(config_path)

    def test_invalid_dataset_raises_with_name(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.json"
        data = {
            "datasets": [
                {
                    "name": "bad_ds",
                    "path": "/tmp",
                    "type": "invalid_type",
                    "basename_regex": ".*",
                    "id_regex": "(.*)",
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="bad_ds"):
            Config.from_file(config_path)

    def test_workdir_prepends_to_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.json"
        data = {
            "datasets": [
                {
                    "name": "test",
                    "path": "relative/data",
                    "type": "rgb",
                    "basename_regex": r"^(?P<f>.+)\.png$",
                    "id_regex": r"^(?P<f>.+)\.png$",
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path, workdir="/work")
        assert config.datasets[0].path == "/work/relative/data"

    def test_workdir_none_leaves_path_unchanged(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.json"
        data = {
            "datasets": [
                {
                    "name": "test",
                    "path": "/absolute/data",
                    "type": "rgb",
                    "basename_regex": r"^(?P<f>.+)\.png$",
                    "id_regex": r"^(?P<f>.+)\.png$",
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path, workdir=None)
        assert config.datasets[0].path == "/absolute/data"

    def test_loads_multiple_datasets(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.json"
        ds_base = {
            "type": "rgb",
            "basename_regex": r"^(?P<f>.+)\.png$",
            "id_regex": r"^(?P<f>.+)\.png$",
        }
        data = {
            "datasets": [
                {"name": "ds1", "path": "/a", **ds_base},
                {"name": "ds2", "path": "/b", **ds_base},
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path)
        assert len(config.datasets) == 2
        assert config.datasets[0].name == "ds1"
        assert config.datasets[1].name == "ds2"

    def test_optional_fields_default(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.json"
        data = {
            "datasets": [
                {
                    "name": "minimal",
                    "path": "/data",
                    "type": "rgb",
                    "basename_regex": r"^(?P<f>.+)\.png$",
                    "id_regex": r"^(?P<f>.+)\.png$",
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path)
        ds = config.datasets[0]
        assert ds.path_regex is None
        assert ds.hierarchy_regex is None
        assert ds.named_capture_group_value_separator is None
        assert ds.intrinsics_regex is None
        assert ds.extrinsics_regex is None
        assert ds.flat_ids_unique is False
        assert ds.id_regex_join_char == "+"
        assert ds.properties == {}
        assert ds.output_json is None
        assert ds.file_extensions is None

    def test_properties_loaded(self, tmp_path: Path) -> None:
        config_path = tmp_path / "cfg.json"
        data = {
            "datasets": [
                {
                    "name": "test",
                    "path": "/data",
                    "type": "depth",
                    "basename_regex": r"^(?P<f>.+)\.png$",
                    "id_regex": r"^(?P<f>.+)\.png$",
                    "properties": {"gt": False, "model": "MyModel"},
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path)
        assert config.datasets[0].properties == {"gt": False, "model": "MyModel"}


# ===================================================================
# Regex matching smoke tests (verify compiled regexes work)
# ===================================================================


class TestRegexPatternMatching:
    """Verify that the regex patterns from the example config actually
    match the expected file paths."""

    def test_vkitti2_basename_matches(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        m = vk.compiled_basename_regex.match("rgb_00001.jpg")
        assert m is not None
        assert m.group("frame") == "00001"
        assert m.group("ext") == "jpg"

    def test_vkitti2_basename_rejects_wrong_format(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        assert vk.compiled_basename_regex.match("depth_00001.png") is None

    def test_vkitti2_id_regex_matches(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        path = "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"
        m = vk.compiled_id_regex.match(path)
        assert m is not None
        assert m.group("scene") == "Scene01"
        assert m.group("variation") == "clone"
        assert m.group("camera") == "Camera_0"
        assert m.group("frame") == "00001"

    def test_vkitti2_id_regex_rejects_wrong_path(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        assert vk.compiled_id_regex.match("wrong/path/file.jpg") is None

    def test_vkitti2_intrinsics_regex_matches(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        path = "Scene01/clone/intrinsics/Camera_0_intrinsics.txt"
        m = vk.compiled_intrinsics_regex.match(path)
        assert m is not None
        assert m.group("scene") == "Scene01"
        assert m.group("variation") == "clone"
        assert m.group("camera") == "Camera_0"

    def test_vkitti2_extrinsics_regex_matches(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        vk = config.datasets[0]
        path = "Scene01/clone/extrinsics/Camera_0_extrinsics.txt"
        m = vk.compiled_extrinsics_regex.match(path)
        assert m is not None

    def test_ddad_id_regex_matches(self, example_config_path: Path) -> None:
        config = Config.from_file(example_config_path)
        ddad = config.datasets[1]
        path = "000150/rgb/CAMERA_01/0000000050.png"
        m = ddad.compiled_id_regex.match(path)
        assert m is not None
        assert m.group("scene") == "000150"
        assert m.group("camera") == "CAMERA_01"
        assert m.group("frame") == "0000000050"

    def test_ddad_intrinsics_regex_matches(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        ddad = config.datasets[1]
        path = "000150/calibration/CAMERA_01.json"
        m = ddad.compiled_intrinsics_regex.match(path)
        assert m is not None
        assert m.group("scene") == "000150"
        assert m.group("camera") == "CAMERA_01"

    def test_depth_predictions_id_regex_matches(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        dp = config.datasets[2]
        path = "Scene01/00001_pred.png"
        m = dp.compiled_id_regex.match(path)
        assert m is not None
        assert m.group("scene") == "Scene01"
        assert m.group("frame") == "00001"
        assert m.group("ext") == "png"

    def test_depth_predictions_npy_matches(
        self, example_config_path: Path
    ) -> None:
        config = Config.from_file(example_config_path)
        dp = config.datasets[2]
        path = "Scene01/00001_pred.npy"
        m = dp.compiled_id_regex.match(path)
        assert m is not None
        assert m.group("ext") == "npy"
