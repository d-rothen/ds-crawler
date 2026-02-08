"""Tests for crawler.config – loading, validation, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_crawler.config import Config, DatasetConfig


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
        from .conftest import default_meta_for_modality

        defaults = {
            "name": "test",
            "path": "/tmp/test",
            "type": "rgb",
            "basename_regex": r"^(?P<name>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<name>.+)\.png$",
        }
        defaults.update(overrides)

        props = defaults.get("properties")
        if props is None:
            props = {}
        if isinstance(props, dict) and "euler_train" not in props:
            props = props.copy()
            modality = defaults["type"]
            props["euler_train"] = {
                "used_as": "input",
                "modality_type": modality,
            }
            meta = default_meta_for_modality(modality)
            if meta is not None and "meta" not in props:
                props["meta"] = meta
            defaults["properties"] = props
        return defaults

    def test_valid_minimal_config(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs())
        assert ds.name == "test"
        assert ds.type == "rgb"

    def test_properties_reserved_type_key_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved top-level key"):
            DatasetConfig(**self._minimal_kwargs(properties={"type": "rgb"}))

    def test_euler_train_required_when_omitted(self) -> None:
        with pytest.raises(ValueError, match="properties.euler_train is required"):
            DatasetConfig(
                name="test",
                path="/tmp/test",
                type="rgb",
                basename_regex=r"^(?P<name>.+)\.(?P<ext>png)$",
                id_regex=r"^(?P<name>.+)\.png$",
                properties={},
            )

    def test_euler_train_must_be_object_when_present(self) -> None:
        with pytest.raises(ValueError, match="properties.euler_train must be an object"):
            DatasetConfig(**self._minimal_kwargs(properties={"euler_train": "bad"}))

    def test_euler_train_unknown_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="Unknown properties.euler_train key"):
            DatasetConfig(**self._minimal_kwargs(properties={"euler_train": {"foo": "bar"}}))

    def test_euler_train_requires_used_as(self) -> None:
        with pytest.raises(ValueError, match="properties.euler_train.used_as is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={"euler_train": {"modality_type": "rgb"}}
                )
            )

    def test_euler_train_requires_modality_type(self) -> None:
        with pytest.raises(
            ValueError, match="properties.euler_train.modality_type is required"
        ):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={"euler_train": {"used_as": "input"}}
                )
            )

    def test_euler_train_used_as_condition_requires_condition_fields_shape(self) -> None:
        with pytest.raises(ValueError, match="applies_to must be a list"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "condition",
                            "modality_type": "metadata",
                            "applies_to": "rgb",
                        }
                    }
                )
            )

    def test_euler_train_slot_is_inferred_when_omitted(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "target",
                        "modality_type": "rgb",
                    },
                    "meta": {
                        "rgb_range": [0, 255],
                    },
                }
            )
        )
        assert ds.euler_train["used_as"] == "target"
        assert ds.euler_train["modality_type"] == "rgb"
        assert ds.euler_train["slot"] == "test.target.rgb"

    def test_euler_train_metadata_condition_defaults_include_condition_fields(
        self,
    ) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                type="metadata",
                hierarchy_regex=r"^(?P<scene>[^/]+)/(?P<camera>[^/]+)/(?P<frame>\d+)\.png$",
                named_capture_group_value_separator=":",
                properties={
                    "euler_train": {
                        "used_as": "condition",
                        "modality_type": "metadata",
                    }
                },
            )
        )
        assert ds.euler_train["used_as"] == "condition"
        assert ds.euler_train["hierarchy_scope"] == "scene_camera_frame"
        assert ds.euler_train["applies_to"] == ["*"]

    def test_depth_modality_requires_meta(self) -> None:
        with pytest.raises(ValueError, match="properties.meta is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        }
                    }
                )
            )

    def test_depth_modality_meta_missing_radial_depth(self) -> None:
        with pytest.raises(ValueError, match="meta.radial_depth is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        },
                        "meta": {"scale_to_meters": 1.0},
                    }
                )
            )

    def test_depth_modality_meta_missing_scale_to_meters(self) -> None:
        with pytest.raises(ValueError, match="meta.scale_to_meters is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        },
                        "meta": {"radial_depth": True},
                    }
                )
            )

    def test_depth_modality_meta_wrong_type_radial_depth(self) -> None:
        with pytest.raises(ValueError, match="meta.radial_depth must be a bool"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        },
                        "meta": {"radial_depth": "yes", "scale_to_meters": 1.0},
                    }
                )
            )

    def test_depth_modality_meta_wrong_type_scale_to_meters(self) -> None:
        with pytest.raises(ValueError, match="meta.scale_to_meters must be a number"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        },
                        "meta": {"radial_depth": False, "scale_to_meters": "one"},
                    }
                )
            )

    def test_depth_modality_meta_valid(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "depth",
                    },
                    "meta": {"radial_depth": True, "scale_to_meters": 0.001},
                }
            )
        )
        assert ds.properties["meta"]["radial_depth"] is True
        assert ds.properties["meta"]["scale_to_meters"] == 0.001

    def test_depth_modality_meta_accepts_int_scale(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "depth",
                    },
                    "meta": {"radial_depth": False, "scale_to_meters": 1},
                }
            )
        )
        assert ds.properties["meta"]["scale_to_meters"] == 1

    def test_unknown_modality_does_not_require_meta(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "flow",
                    }
                }
            )
        )
        assert "meta" not in ds.properties

    def test_semseg_modality_requires_meta(self) -> None:
        with pytest.raises(ValueError, match="properties.meta is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "semantic_segmentation",
                        }
                    }
                )
            )

    def test_semseg_modality_meta_missing_skyclass(self) -> None:
        with pytest.raises(ValueError, match="meta.skyclass is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "semantic_segmentation",
                        },
                        "meta": {},
                    }
                )
            )

    def test_semseg_skyclass_not_a_list(self) -> None:
        with pytest.raises(ValueError, match="meta.skyclass must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "semantic_segmentation",
                        },
                        "meta": {"skyclass": "red"},
                    }
                )
            )

    def test_semseg_skyclass_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="meta.skyclass must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "semantic_segmentation",
                        },
                        "meta": {"skyclass": [0, 128]},
                    }
                )
            )

    def test_semseg_skyclass_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="meta.skyclass must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "semantic_segmentation",
                        },
                        "meta": {"skyclass": [0, 128, 300]},
                    }
                )
            )

    def test_semseg_skyclass_valid(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "semantic_segmentation",
                    },
                    "meta": {"skyclass": [135, 206, 235]},
                }
            )
        )
        assert ds.properties["meta"]["skyclass"] == [135, 206, 235]

    def test_rgb_modality_requires_meta(self) -> None:
        with pytest.raises(ValueError, match="properties.meta is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        }
                    }
                )
            )

    def test_rgb_modality_meta_missing_rgb_range(self) -> None:
        with pytest.raises(ValueError, match="meta.rgb_range is required"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {},
                    }
                )
            )

    def test_rgb_modality_meta_wrong_type(self) -> None:
        with pytest.raises(ValueError, match="meta.rgb_range must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {"rgb_range": "0-255"},
                    }
                )
            )

    def test_rgb_modality_meta_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="meta.rgb_range must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {"rgb_range": [0, 128, 255]},
                    }
                )
            )

    def test_rgb_modality_meta_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match="meta.rgb_range must be"):
            DatasetConfig(
                **self._minimal_kwargs(
                    properties={
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {"rgb_range": [255, 0]},
                    }
                )
            )

    def test_rgb_modality_meta_valid(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "rgb",
                    },
                    "meta": {"rgb_range": [0, 255]},
                }
            )
        )
        assert ds.properties["meta"]["rgb_range"] == [0, 255]

    def test_rgb_modality_meta_normalised_range(self) -> None:
        ds = DatasetConfig(
            **self._minimal_kwargs(
                properties={
                    "euler_train": {
                        "used_as": "input",
                        "modality_type": "rgb",
                    },
                    "meta": {"rgb_range": [0, 1]},
                }
            )
        )
        assert ds.properties["meta"]["rgb_range"] == [0, 1]

    def test_id_override_accepted(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(id_override="calibration"))
        assert ds.id_override == "calibration"

    def test_id_override_default_none(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs())
        assert ds.id_override is None

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
            "properties": {
                "euler_train": {
                    "used_as": "input",
                    "modality_type": "rgb",
                },
                "meta": {
                    "rgb_range": [0, 255],
                },
            },
        }
        defaults.update(overrides)
        props = defaults.get("properties")
        if isinstance(props, dict) and "euler_train" not in props:
            props = props.copy()
            modality = defaults["type"]
            props["euler_train"] = {
                "used_as": "input",
                "modality_type": modality,
            }
            from .conftest import default_meta_for_modality
            meta = default_meta_for_modality(modality)
            if meta is not None and "meta" not in props:
                props["meta"] = meta
            defaults["properties"] = props
        return defaults

    def test_no_extensions_returns_none(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="rgb"))
        assert ds.get_file_extensions() is None

    def test_no_extensions_returns_none_depth(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="depth"))
        assert ds.get_file_extensions() is None

    def test_no_extensions_returns_none_segmentation(self) -> None:
        ds = DatasetConfig(**self._minimal_kwargs(type="segmentation"))
        assert ds.get_file_extensions() is None

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
                    "properties": {
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {
                            "rgb_range": [0, 255],
                        },
                    },
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
                    "properties": {
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {
                            "rgb_range": [0, 255],
                        },
                    },
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
            "properties": {
                "euler_train": {
                    "used_as": "input",
                    "modality_type": "rgb",
                },
                "meta": {
                    "rgb_range": [0, 255],
                },
            },
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
                    "properties": {
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "rgb",
                        },
                        "meta": {
                            "rgb_range": [0, 255],
                        },
                    },
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
        assert ds.properties == {
            "euler_train": {
                "used_as": "input",
                "modality_type": "rgb",
            },
            "meta": {
                "rgb_range": [0, 255],
            },
        }
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
                    "properties": {
                        "gt": False,
                        "model": "MyModel",
                        "euler_train": {
                            "used_as": "input",
                            "modality_type": "depth",
                        },
                        "meta": {
                            "radial_depth": False,
                            "scale_to_meters": 1.0,
                        },
                    },
                }
            ]
        }
        config_path.write_text(json.dumps(data))
        config = Config.from_file(config_path)
        assert config.datasets[0].properties == {
            "gt": False,
            "model": "MyModel",
            "euler_train": {
                "used_as": "input",
                "modality_type": "depth",
            },
            "meta": {
                "radial_depth": False,
                "scale_to_meters": 1.0,
            },
        }


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
