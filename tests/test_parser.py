"""Tests for crawler.parser – helper functions and DatasetParser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.config import Config, DatasetConfig
from ds_crawler.parser import (
    DatasetParser,
    index_dataset,
    index_dataset_from_files,
    _add_file_to_node,
    _deep_merge,
    _ensure_hierarchy_path,
    _get_hierarchy_keys,
)
from ds_crawler.traversal import (
    get_files,
    _collect_all_referenced_paths,
    _collect_ids,
)

from .conftest import (
    create_vkitti2_tree,
    create_depth_predictions_tree,
    make_vkitti2_config,
    make_ddad_config,
    make_depth_predictions_config,
    touch,
    write_config_json,
)


def make_dataset_config(**kwargs: Any) -> DatasetConfig:
    """Build a DatasetConfig with minimal euler_train defaults for parser tests."""
    props = kwargs.get("properties")
    if props is None:
        props = {}
    if isinstance(props, dict) and "euler_train" not in props:
        props = props.copy()
        modality = kwargs.get("type", "rgb")
        props["euler_train"] = {
            "used_as": "input",
            "modality_type": modality,
        }
        if modality == "depth" and "meta" not in props:
            props["meta"] = {"radial_depth": False, "scale_to_meters": 1.0}
    kwargs["properties"] = props
    return DatasetConfig(**kwargs)


def with_euler_train(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure a plain config dict has minimal required euler_train metadata."""
    result = config.copy()
    props = result.get("properties")
    if props is None:
        props = {}
    if isinstance(props, dict) and "euler_train" not in props:
        props = props.copy()
        modality_type = str(result.get("type") or "unknown")
        used_as = "condition" if modality_type == "metadata" else "input"
        props["euler_train"] = {
            "used_as": used_as,
            "modality_type": modality_type,
        }
        if modality_type == "depth" and "meta" not in props:
            props["meta"] = {"radial_depth": False, "scale_to_meters": 1.0}
        result["properties"] = props
    return result


# ===================================================================
# _deep_merge
# ===================================================================


class TestDeepMerge:
    def test_empty_dicts(self) -> None:
        assert _deep_merge({}, {}) == {}

    def test_override_adds_keys(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_replaces_values(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_recursive_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_non_dict_replaces_dict(self) -> None:
        base = {"x": {"a": 1}}
        override = {"x": "replaced"}
        assert _deep_merge(base, override) == {"x": "replaced"}

    def test_dict_replaces_non_dict(self) -> None:
        base = {"x": "scalar"}
        override = {"x": {"a": 1}}
        assert _deep_merge(base, override) == {"x": {"a": 1}}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}

    def test_deeply_nested(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 3, "e": 4}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 3, "e": 4}}}


# ===================================================================
# _get_hierarchy_keys
# ===================================================================


class TestGetHierarchyKeys:
    def test_named_groups_with_separator(self) -> None:
        pattern = re.compile(r"^(?P<scene>Scene\d+)/(?P<camera>Cam\d+)$")
        m = pattern.match("Scene01/Cam0")
        keys = _get_hierarchy_keys(m, ":")
        assert keys == ["scene:Scene01", "camera:Cam0"]

    def test_named_groups_different_separator(self) -> None:
        pattern = re.compile(r"^(?P<a>x)/(?P<b>y)$")
        m = pattern.match("x/y")
        keys = _get_hierarchy_keys(m, "-")
        assert keys == ["a-x", "b-y"]

    def test_unnamed_groups(self) -> None:
        pattern = re.compile(r"^(Scene\d+)/(Cam\d+)$")
        m = pattern.match("Scene01/Cam0")
        keys = _get_hierarchy_keys(m, None)
        assert keys == ["Scene01", "Cam0"]

    def test_optional_group_not_matched(self) -> None:
        pattern = re.compile(r"^(?P<scene>Scene\d+)(?:/(?P<sub>\w+))?$")
        m = pattern.match("Scene01")
        keys = _get_hierarchy_keys(m, ":")
        # "sub" didn't match, so only scene is in keys
        assert keys == ["scene:Scene01"]

    def test_single_named_group(self) -> None:
        pattern = re.compile(r"^(?P<frame>\d+)\.png$")
        m = pattern.match("00001.png")
        keys = _get_hierarchy_keys(m, ":")
        assert keys == ["frame:00001"]

    def test_single_unnamed_group(self) -> None:
        pattern = re.compile(r"^(\d+)\.png$")
        m = pattern.match("00001.png")
        keys = _get_hierarchy_keys(m, None)
        assert keys == ["00001"]


# ===================================================================
# _ensure_hierarchy_path
# ===================================================================


class TestEnsureHierarchyPath:
    def test_empty_keys_returns_root(self) -> None:
        root: dict = {}
        node = _ensure_hierarchy_path(root, [])
        assert node is root

    def test_single_key(self) -> None:
        root: dict = {}
        node = _ensure_hierarchy_path(root, ["scene:A"])
        assert "children" in root
        assert "scene:A" in root["children"]
        assert node is root["children"]["scene:A"]
        assert node == {}

    def test_multiple_keys(self) -> None:
        root: dict = {}
        node = _ensure_hierarchy_path(root, ["a", "b", "c"])
        assert node == {}
        # Verify full path exists
        assert root["children"]["a"]["children"]["b"]["children"]["c"] is node

    def test_idempotent(self) -> None:
        root: dict = {}
        node1 = _ensure_hierarchy_path(root, ["a", "b"])
        node2 = _ensure_hierarchy_path(root, ["a", "b"])
        assert node1 is node2

    def test_branching(self) -> None:
        root: dict = {}
        _ensure_hierarchy_path(root, ["a", "x"])
        _ensure_hierarchy_path(root, ["a", "y"])
        assert set(root["children"]["a"]["children"].keys()) == {"x", "y"}

    def test_preserves_existing_data(self) -> None:
        root: dict = {"children": {"a": {"existing_key": "value"}}}
        node = _ensure_hierarchy_path(root, ["a", "b"])
        assert root["children"]["a"]["existing_key"] == "value"
        assert node == {}


# ===================================================================
# _add_file_to_node
# ===================================================================


class TestAddFileToNode:
    def test_creates_files_list(self) -> None:
        node: dict = {}
        entry = {"path": "a.png", "id": "a"}
        _add_file_to_node(node, entry)
        assert node == {"files": [entry]}

    def test_appends_to_existing(self) -> None:
        entry1 = {"path": "a.png", "id": "a"}
        entry2 = {"path": "b.png", "id": "b"}
        node: dict = {"files": [entry1]}
        _add_file_to_node(node, entry2)
        assert node["files"] == [entry1, entry2]

    def test_preserves_other_keys(self) -> None:
        node: dict = {"camera_intrinsics": "path/to/intrinsics.txt"}
        entry = {"path": "a.png", "id": "a"}
        _add_file_to_node(node, entry)
        assert node["camera_intrinsics"] == "path/to/intrinsics.txt"
        assert len(node["files"]) == 1


# ===================================================================
# DatasetParser._process_file
# ===================================================================


class TestProcessFile:
    """Test the _process_file method with various configs."""

    def _make_parser(self, ds_config: DatasetConfig) -> DatasetParser:
        config = Config(datasets=[ds_config])
        return DatasetParser(config)

    def test_vkitti2_file(self, vkitti2_dataset_config: DatasetConfig) -> None:
        parser = self._make_parser(vkitti2_dataset_config)
        base = Path(vkitti2_dataset_config.path)
        file_path = base / "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"

        entry, skip = parser._process_file(file_path, base, vkitti2_dataset_config)
        assert skip is None
        assert entry is not None
        assert entry["path"] == "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"
        assert entry["id"] == "scene-Scene01+variation-clone+camera-Camera_0+frame-00001"
        assert entry["basename_properties"] == {"frame": "00001", "ext": "jpg"}
        assert entry["path_properties"] == {
            "scene": "Scene01",
            "variation": "clone",
            "camera": "Camera_0",
        }

    def test_ddad_file(self, ddad_dataset_config: DatasetConfig) -> None:
        parser = self._make_parser(ddad_dataset_config)
        base = Path(ddad_dataset_config.path)
        file_path = base / "000150/rgb/CAMERA_01/0000000050.png"

        entry, skip = parser._process_file(file_path, base, ddad_dataset_config)
        assert skip is None
        assert entry is not None
        assert entry["id"] == "scene-000150+camera-CAMERA_01+frame-0000000050"
        assert entry["path_properties"] == {"scene": "000150", "camera": "CAMERA_01"}

    def test_depth_predictions_file(
        self, depth_predictions_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(depth_predictions_dataset_config)
        base = Path(depth_predictions_dataset_config.path)
        file_path = base / "Scene01/00001_pred.png"

        entry, skip = parser._process_file(
            file_path, base, depth_predictions_dataset_config
        )
        assert skip is None
        assert entry is not None
        assert entry["id"] == "scene-Scene01+frame-00001+ext-png"
        assert entry["path_properties"] == {}
        assert entry["basename_properties"] == {"frame": "00001", "ext": "png"}

    def test_depth_predictions_npy(
        self, depth_predictions_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(depth_predictions_dataset_config)
        base = Path(depth_predictions_dataset_config.path)
        file_path = base / "Scene01/00001_pred.npy"

        entry, skip = parser._process_file(
            file_path, base, depth_predictions_dataset_config
        )
        assert entry is not None
        assert entry["id"] == "scene-Scene01+frame-00001+ext-npy"
        assert entry["basename_properties"]["ext"] == "npy"

    def test_skip_basename_mismatch(
        self, vkitti2_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(vkitti2_dataset_config)
        base = Path(vkitti2_dataset_config.path)
        file_path = base / "Scene01/clone/intrinsics/Camera_0_intrinsics.txt"

        entry, skip = parser._process_file(file_path, base, vkitti2_dataset_config)
        assert entry is None
        assert skip == "basename"

    def test_skip_id_regex_mismatch(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^subdir/(?P<f>.+)\.png$",
        )
        parser = self._make_parser(ds)
        # File at root level won't match id_regex that requires "subdir/"
        file_path = tmp_path / "file.png"
        touch(file_path)

        entry, skip = parser._process_file(file_path, tmp_path, ds)
        assert entry is None
        assert skip == "id_regex"

    def test_skip_path_regex_mismatch(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<f>.+)\.png$",
            path_regex=r"^specific_dir/",
        )
        parser = self._make_parser(ds)
        file_path = tmp_path / "other_dir" / "file.png"
        touch(file_path)

        entry, skip = parser._process_file(file_path, tmp_path, ds)
        assert entry is None
        assert skip == "path_regex"

    def test_unnamed_id_regex_groups(self, tmp_path: Path) -> None:
        """Test ID construction with unnamed capture groups."""
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(.+)\.(?P<ext>png)$",
            id_regex=r"^([^/]+)/(\d+)\.png$",
            id_regex_join_char="-",
        )
        parser = self._make_parser(ds)
        file_path = tmp_path / "scene01" / "00001.png"
        touch(file_path)

        entry, skip = parser._process_file(file_path, tmp_path, ds)
        assert entry is not None
        assert entry["id"] == "scene01-00001"

    def test_id_override_replaces_extracted_id(self, tmp_path: Path) -> None:
        """When id_override is set, the extracted ID is replaced."""
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="calibration",
            id_regex=r"^.*/(\w+)\.json$",
            id_override="intrinsics",
        )
        parser = self._make_parser(ds)
        file_path = tmp_path / "scene01" / "d0b061d777c48f07f1ba3c51fa5a80451745b345.json"
        touch(file_path)

        entry, skip = parser._process_file(file_path, tmp_path, ds)
        assert skip is None
        assert entry is not None
        assert entry["id"] == "intrinsics"

    def test_id_override_none_uses_extracted_id(self, tmp_path: Path) -> None:
        """When id_override is None, the regex-extracted ID is used."""
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="calibration",
            id_regex=r"^.*/(\w+)\.json$",
        )
        parser = self._make_parser(ds)
        file_path = tmp_path / "scene01" / "some_uuid.json"
        touch(file_path)

        entry, skip = parser._process_file(file_path, tmp_path, ds)
        assert skip is None
        assert entry is not None
        assert entry["id"] == "some_uuid"


# ===================================================================
# DatasetParser._get_entry_hierarchy_keys
# ===================================================================


class TestGetEntryHierarchyKeys:
    def _make_parser(self, ds_config: DatasetConfig) -> DatasetParser:
        config = Config(datasets=[ds_config])
        return DatasetParser(config)

    def test_no_hierarchy_regex_returns_empty(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        parser = self._make_parser(ds)
        keys = parser._get_entry_hierarchy_keys("file.png", ds)
        assert keys == []

    def test_returns_hierarchy_keys(
        self, vkitti2_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(vkitti2_dataset_config)
        path = "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"
        keys = parser._get_entry_hierarchy_keys(path, vkitti2_dataset_config)
        assert keys == [
            "scene:Scene01",
            "variation:clone",
            "camera:Camera_0",
            "frame:00001",
        ]

    def test_returns_none_on_mismatch(
        self, vkitti2_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(vkitti2_dataset_config)
        keys = parser._get_entry_hierarchy_keys(
            "wrong/path.jpg", vkitti2_dataset_config
        )
        assert keys is None

    def test_ddad_hierarchy_keys(
        self, ddad_dataset_config: DatasetConfig
    ) -> None:
        parser = self._make_parser(ddad_dataset_config)
        keys = parser._get_entry_hierarchy_keys(
            "000150/rgb/CAMERA_01/0000000050.png", ddad_dataset_config
        )
        assert keys == ["scene:000150", "camera:CAMERA_01", "frame:0000000050"]


# ===================================================================
# DatasetParser._build_output
# ===================================================================


class TestBuildOutput:
    def _make_parser(self, ds_config: DatasetConfig) -> DatasetParser:
        config = Config(datasets=[ds_config])
        return DatasetParser(config)

    def test_basic_output_structure(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="MyDS",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
            id_regex_join_char="+",
            properties={
                "gt": True,
                "euler_train": {
                    "used_as": "target",
                    "slot": "demo.target.rgb",
                    "modality_type": "rgb",
                },
            },
        )
        parser = self._make_parser(ds)
        node: dict = {"children": {"a": {"files": []}}}
        output = parser._build_output(ds, node)

        assert output["name"] == "MyDS"
        assert output["id_regex"] == r"^(?P<f>.+)\.png$"
        assert output["id_regex_join_char"] == "+"
        assert output["type"] == "rgb"
        assert output["euler_train"] == {
            "used_as": "target",
            "slot": "demo.target.rgb",
            "modality_type": "rgb",
        }
        assert output["gt"] is True
        assert "dataset" in output
        assert output["dataset"]["children"]["a"]["files"] == []

    def test_hierarchy_regex_included(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
            hierarchy_regex=r"^(?P<f>.+)\.png$",
            named_capture_group_value_separator=":",
        )
        parser = self._make_parser(ds)
        output = parser._build_output(ds, {})

        assert output["hierarchy_regex"] == r"^(?P<f>.+)\.png$"
        assert output["named_capture_group_value_separator"] == ":"

    def test_no_hierarchy_regex_omitted(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        parser = self._make_parser(ds)
        output = parser._build_output(ds, {})

        assert "hierarchy_regex" not in output
        assert "named_capture_group_value_separator" not in output

    def test_id_override_included_in_output(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="calibration",
            id_regex=r"^(?P<f>.+)\.json$",
            id_override="intrinsics",
        )
        parser = self._make_parser(ds)
        output = parser._build_output(ds, {})
        assert output["id_override"] == "intrinsics"

    def test_id_override_none_omitted_from_output(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        parser = self._make_parser(ds)
        output = parser._build_output(ds, {})
        assert "id_override" not in output

    def test_dataset_properties_deep_merged(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
            properties={
                "dataset": {
                    "license": "MIT",
                    "source": "http://example.com",
                },
            },
        )
        parser = self._make_parser(ds)
        dataset_node: dict = {"children": {"a": {}}}
        output = parser._build_output(ds, dataset_node)

        # dataset properties should be merged into the dataset node
        assert output["dataset"]["license"] == "MIT"
        assert output["dataset"]["source"] == "http://example.com"
        assert "children" in output["dataset"]
        # "dataset" key from properties should NOT leak to top level
        assert "license" not in output
        assert "source" not in output


# ===================================================================
# DatasetParser.parse_dataset – with real filesystem
# ===================================================================


class TestParseDataset:
    def test_vkitti2_parse(self, vkitti2_root: Path) -> None:
        cfg = make_vkitti2_config(str(vkitti2_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        # Should have children at top level
        assert "children" in dataset_node

        # Check Scene01 exists
        assert "scene:Scene01" in dataset_node["children"]
        scene01 = dataset_node["children"]["scene:Scene01"]

        # Check variation:clone exists
        assert "variation:clone" in scene01["children"]
        clone = scene01["children"]["variation:clone"]

        # Check Camera_0 exists
        assert "camera:Camera_0" in clone["children"]
        cam0 = clone["children"]["camera:Camera_0"]

        # Check camera files indexed
        assert cam0["camera_intrinsics"] == "Scene01/clone/intrinsics/Camera_0_intrinsics.txt"
        assert cam0["camera_extrinsics"] == "Scene01/clone/extrinsics/Camera_0_extrinsics.txt"

        # Check frame files
        assert "frame:00001" in cam0["children"]
        assert "frame:00002" in cam0["children"]
        frame1_files = cam0["children"]["frame:00001"]["files"]
        assert len(frame1_files) == 1
        assert frame1_files[0]["id"] == "scene-Scene01+variation-clone+camera-Camera_0+frame-00001"

        # Check Camera_1
        cam1 = clone["children"]["camera:Camera_1"]
        assert cam1["camera_intrinsics"] == "Scene01/clone/intrinsics/Camera_1_intrinsics.txt"
        assert "frame:00001" in cam1["children"]

        # Check Scene02
        assert "scene:Scene02" in dataset_node["children"]
        scene02 = dataset_node["children"]["scene:Scene02"]
        fog_cam0 = scene02["children"]["variation:fog"]["children"]["camera:Camera_0"]
        assert fog_cam0["camera_intrinsics"] == "Scene02/fog/intrinsics/Camera_0_intrinsics.txt"

    def test_ddad_parse(self, ddad_root: Path) -> None:
        cfg = make_ddad_config(str(ddad_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        scene = dataset_node["children"]["scene:000150"]
        cam01 = scene["children"]["camera:CAMERA_01"]
        assert cam01["camera_intrinsics"] == "000150/calibration/CAMERA_01.json"

        # Check both frames
        assert "frame:0000000050" in cam01["children"]
        assert "frame:0000000051" in cam01["children"]

        cam05 = scene["children"]["camera:CAMERA_05"]
        assert cam05["camera_intrinsics"] == "000150/calibration/CAMERA_05.json"
        assert "frame:0000000050" in cam05["children"]

    def test_depth_predictions_parse(self, depth_predictions_root: Path) -> None:
        cfg = make_depth_predictions_config(str(depth_predictions_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        scene = dataset_node["children"]["scene:Scene01"]

        # frame:00001 should have both .png and .npy files
        frame01 = scene["children"]["frame:00001"]
        assert len(frame01["files"]) == 2
        exts = {f["basename_properties"]["ext"] for f in frame01["files"]}
        assert exts == {"png", "npy"}

        # frame:00002 should also have both
        frame02 = scene["children"]["frame:00002"]
        assert len(frame02["files"]) == 2

    def test_flat_structure_without_hierarchy_regex(self, tmp_path: Path) -> None:
        """Without hierarchy_regex, files go into a flat structure."""
        root = tmp_path / "data"
        touch(root / "file1.png")
        touch(root / "file2.png")

        ds = make_dataset_config(
            name="flat",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<name>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<name>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)

        # Without hierarchy_regex, files are placed at root with empty keys
        assert "files" in node
        assert len(node["files"]) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        ds = make_dataset_config(
            name="empty",
            path=str(root),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)
        assert node == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        ds = make_dataset_config(
            name="missing",
            path=str(tmp_path / "nonexistent"),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)
        assert node == {}


# ===================================================================
# Duplicate detection
# ===================================================================


class TestDuplicateDetection:
    def test_flat_ids_unique_skips_duplicates(self, tmp_path: Path) -> None:
        """Global duplicate detection: same ID from different paths is skipped."""
        root = tmp_path / "data"
        touch(root / "a" / "001.png")
        touch(root / "b" / "001.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>\d+)\.(?P<ext>png)$",
            # id_regex produces same ID for both files (only captures filename)
            id_regex=r"^[^/]+/(?P<frame>\d+)\.png$",
            hierarchy_regex=r"^([^/]+)/(\d+)\.png$",
            flat_ids_unique=True,
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)

        # Both dirs match, but IDs are the same – only one should be kept
        all_files = self._collect_all_files(node)
        assert len(all_files) == 1

    def test_per_level_duplicates_allowed_across_levels(
        self, tmp_path: Path
    ) -> None:
        """Per-level duplicate detection: same ID in different hierarchy levels is fine."""
        root = tmp_path / "data"
        touch(root / "a" / "001.png")
        touch(root / "b" / "001.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>\d+)\.(?P<ext>png)$",
            id_regex=r"^[^/]+/(?P<frame>\d+)\.png$",
            hierarchy_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
            named_capture_group_value_separator=":",
            flat_ids_unique=False,
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)

        # Different hierarchy levels: dir:a/frame:001 vs dir:b/frame:001
        # These are different levels, so both entries are kept
        all_files = self._collect_all_files(node)
        assert len(all_files) == 2

    def test_per_level_duplicate_same_level_skipped(
        self, tmp_path: Path
    ) -> None:
        """Per-level: exact same hierarchy key path with same ID is still a duplicate."""
        root = tmp_path / "data"
        # Two files that produce the same hierarchy keys AND same ID
        touch(root / "scene" / "001.png")
        touch(root / "scene" / "001.jpg")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png", ".jpg"],
            basename_regex=r"^(?P<f>\d+)\.(?P<ext>png|jpg)$",
            # id_regex doesn't include extension, so both files get same ID
            id_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.(?:png|jpg)$",
            hierarchy_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.(?:png|jpg)$",
            named_capture_group_value_separator=":",
            flat_ids_unique=False,
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        node = parser.parse_dataset(ds)

        all_files = self._collect_all_files(node)
        assert len(all_files) == 1

    def test_strict_mode_raises_on_duplicate(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a" / "001.png")
        touch(root / "b" / "001.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>\d+)\.(?P<ext>png)$",
            id_regex=r"^[^/]+/(?P<frame>\d+)\.png$",
            hierarchy_regex=r"^([^/]+)/(\d+)\.png$",
            flat_ids_unique=True,
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config, strict=True)

        with pytest.raises(RuntimeError, match="Duplicate id"):
            parser.parse_dataset(ds)

    def test_id_override_duplicate_same_level_verbose_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """id_override causing duplicates produces a verbose warning."""
        root = tmp_path / "data"
        touch(root / "scene" / "uuid_a.json")
        touch(root / "scene" / "uuid_b.json")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="calibration",
            file_extensions=[".json"],
            id_regex=r"^([^/]+)/(\w+)\.json$",
            hierarchy_regex=r"^([^/]+)/",
            id_override="intrinsics",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)

        import logging
        with caplog.at_level(logging.WARNING):
            node = parser.parse_dataset(ds)

        all_files = self._collect_all_files(node)
        assert len(all_files) == 1  # second file skipped as duplicate
        assert "id_override" in caplog.text
        assert "only one file per hierarchy level" in caplog.text

    def test_id_override_duplicate_strict_raises_verbose(
        self, tmp_path: Path
    ) -> None:
        """id_override causing duplicates in strict mode raises with verbose message."""
        root = tmp_path / "data"
        touch(root / "scene" / "uuid_a.json")
        touch(root / "scene" / "uuid_b.json")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="calibration",
            file_extensions=[".json"],
            id_regex=r"^([^/]+)/(\w+)\.json$",
            hierarchy_regex=r"^([^/]+)/",
            id_override="intrinsics",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config, strict=True)

        with pytest.raises(RuntimeError, match="id_override"):
            parser.parse_dataset(ds)

    def _collect_all_files(self, node: dict) -> list[dict]:
        """Recursively collect all file entries from a hierarchy."""
        files = list(node.get("files", []))
        for child in node.get("children", {}).values():
            files.extend(self._collect_all_files(child))
        return files


# ===================================================================
# ID miss threshold
# ===================================================================


class TestIdMissThreshold:
    def test_excessive_misses_warns_but_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """High miss ratio logs a warning but never raises, even in strict mode."""
        root = tmp_path / "data"
        for i in range(10):
            touch(root / f"wrong_dir_{i}" / "file.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^correct_dir/(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config, strict=True)

        import logging
        with caplog.at_level(logging.WARNING):
            node = parser.parse_dataset(ds)

        assert node == {}
        assert "failed ID extraction" in caplog.text

    def test_non_strict_does_not_raise(self, tmp_path: Path) -> None:
        """Non-strict mode warns but continues."""
        root = tmp_path / "data"
        for i in range(10):
            touch(root / f"wrong_dir_{i}" / "file.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^correct_dir/(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config, strict=False)

        # Should not raise
        node = parser.parse_dataset(ds)
        assert node == {}


# ===================================================================
# DatasetParser.parse_all
# ===================================================================


class TestParseAll:
    def test_parses_all_datasets(self, full_config: Config) -> None:
        parser = DatasetParser(full_config)
        results = parser.parse_all()

        assert len(results) == 3
        names = [r["name"] for r in results]
        assert names == ["VKITTI2", "DDAD", "depth_predictions"]

    def test_each_result_has_required_fields(self, full_config: Config) -> None:
        parser = DatasetParser(full_config)
        results = parser.parse_all()

        for result in results:
            assert "name" in result
            assert "id_regex" in result
            assert "id_regex_join_char" in result
            assert "dataset" in result


# ===================================================================
# Camera file processing
# ===================================================================


class TestCameraFileProcessing:
    def test_intrinsics_placed_in_hierarchy(self, vkitti2_root: Path) -> None:
        cfg = make_vkitti2_config(str(vkitti2_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        cam0 = (
            dataset_node["children"]["scene:Scene01"]["children"]
            ["variation:clone"]["children"]["camera:Camera_0"]
        )
        assert "camera_intrinsics" in cam0
        assert cam0["camera_intrinsics"] == "Scene01/clone/intrinsics/Camera_0_intrinsics.txt"

    def test_extrinsics_placed_in_hierarchy(self, vkitti2_root: Path) -> None:
        cfg = make_vkitti2_config(str(vkitti2_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        cam0 = (
            dataset_node["children"]["scene:Scene01"]["children"]
            ["variation:clone"]["children"]["camera:Camera_0"]
        )
        assert "camera_extrinsics" in cam0
        assert cam0["camera_extrinsics"] == "Scene01/clone/extrinsics/Camera_0_extrinsics.txt"

    def test_ddad_intrinsics(self, ddad_root: Path) -> None:
        cfg = make_ddad_config(str(ddad_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        dataset_node = parser.parse_dataset(ds_config)

        cam01 = (
            dataset_node["children"]["scene:000150"]["children"]
            ["camera:CAMERA_01"]
        )
        assert cam01["camera_intrinsics"] == "000150/calibration/CAMERA_01.json"

    def test_no_camera_regex_returns_zero(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)

        files = [root / "file.png"]
        count = parser._process_camera_files(
            files, root, ds, {}, "intrinsics"
        )
        assert count == 0


# ===================================================================
# Output writing
# ===================================================================


class TestWriteOutput:
    def test_write_single_output(
        self, full_config: Config, tmp_path: Path
    ) -> None:
        parser = DatasetParser(full_config)
        output_path = tmp_path / "output.json"
        parser.write_output(output_path)

        assert output_path.exists()
        import json

        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_write_per_dataset_output(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        paths = parser.write_outputs_per_dataset()

        assert len(paths) == 1
        assert paths[0] == root / ".ds_crawler" / "output.json"
        assert paths[0].exists()

    def test_write_per_dataset_custom_output_json(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.png")
        custom_output = tmp_path / "custom_output.json"

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<f>.+)\.png$",
            output_json=str(custom_output),
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        paths = parser.write_outputs_per_dataset()

        assert len(paths) == 1
        assert paths[0] == custom_output
        assert custom_output.exists()


# ===================================================================
# parse_dataset_from_files / index_dataset_from_files
# ===================================================================


class TestParseDatasetFromFiles:
    """Verify that passing pre-collected file paths produces the same output
    as letting the handler discover them."""

    def test_matches_handler_output(self, tmp_path: Path) -> None:
        """index_dataset_from_files matches index_dataset for the same files."""
        root = tmp_path / "dp"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))

        # Standard handler-based result
        expected = index_dataset(cfg)

        # Collect the same files manually and use the new API
        from ds_crawler.handlers.generic import GenericHandler

        ds_config = make_dataset_config(**cfg)
        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        result = index_dataset_from_files(cfg, files)
        assert result == expected

    def test_accepts_string_paths(self, tmp_path: Path) -> None:
        """Files can be passed as strings instead of Path objects."""
        root = tmp_path / "dp"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))

        expected = index_dataset(cfg)

        from ds_crawler.handlers.generic import GenericHandler

        ds_config = make_dataset_config(**cfg)
        handler = GenericHandler(ds_config)
        string_files = [str(f) for f in handler.get_files()]

        result = index_dataset_from_files(cfg, string_files)
        assert result == expected

    def test_accepts_iterator(self, tmp_path: Path) -> None:
        """Files can be passed as a lazy iterator."""
        root = tmp_path / "dp"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))

        expected = index_dataset(cfg)

        from ds_crawler.handlers.generic import GenericHandler

        ds_config = make_dataset_config(**cfg)
        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        result = index_dataset_from_files(cfg, iter(files))
        assert result == expected

    def test_custom_base_path(self, tmp_path: Path) -> None:
        """A custom base_path overrides ds_config.path for relative path
        computation."""
        root = tmp_path / "dp"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))

        expected = index_dataset(cfg)

        from ds_crawler.handlers.generic import GenericHandler

        ds_config = make_dataset_config(**cfg)
        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        # Passing the same base_path explicitly should yield the same result
        result = index_dataset_from_files(cfg, files, base_path=str(root))
        assert result == expected

    def test_strict_raises_on_duplicate(self, tmp_path: Path) -> None:
        """strict=True raises RuntimeError on duplicate IDs."""
        root = tmp_path / "data"
        files = [
            touch(root / "a" / "001.png"),
            touch(root / "b" / "001.png"),
        ]
        cfg = with_euler_train({
            "name": "test",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>\d+)\.(?P<ext>png)$",
            "id_regex": r"^[^/]+/(?P<frame>\d+)\.png$",
            "hierarchy_regex": r"^([^/]+)/(\d+)\.png$",
            "flat_ids_unique": True,
        })

        with pytest.raises(RuntimeError, match="Duplicate id"):
            index_dataset_from_files(cfg, files, strict=True)

    def test_vkitti2_matches_handler(self, tmp_path: Path) -> None:
        """Full VKITTI2 dataset with camera files matches handler output."""
        root = tmp_path / "vkitti2"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))

        expected = index_dataset(cfg)

        from ds_crawler.handlers.generic import GenericHandler

        ds_config = make_dataset_config(**cfg)
        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        result = index_dataset_from_files(cfg, files)
        assert result == expected

    def test_empty_file_list(self, tmp_path: Path) -> None:
        """An empty file list produces an empty dataset node."""
        root = tmp_path / "empty"
        root.mkdir()
        cfg = make_depth_predictions_config(str(root))

        result = index_dataset_from_files(cfg, [])
        assert result["dataset"] == {}

    def test_parse_dataset_from_files_method(self, tmp_path: Path) -> None:
        """DatasetParser.parse_dataset_from_files works directly."""
        root = tmp_path / "dp"
        create_depth_predictions_tree(root)
        cfg = make_depth_predictions_config(str(root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])

        parser = DatasetParser(config)
        handler_node = parser.parse_dataset(ds_config)

        from ds_crawler.handlers.generic import GenericHandler

        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        files_node = parser.parse_dataset_from_files(ds_config, files)
        assert files_node == handler_node


# ===================================================================
# _collect_ids
# ===================================================================


class TestCollectIds:
    def test_collects_from_flat_structure(self) -> None:
        output: dict[str, Any] = {
            "dataset": {
                "files": [
                    {"id": "a", "path": "a.png"},
                    {"id": "b", "path": "b.png"},
                ]
            }
        }
        assert _collect_ids(output) == {"a", "b"}

    def test_collects_from_nested_hierarchy(self) -> None:
        output: dict[str, Any] = {
            "dataset": {
                "children": {
                    "level1": {
                        "children": {
                            "level2": {
                                "files": [{"id": "deep", "path": "deep.png"}]
                            }
                        }
                    }
                }
            }
        }
        assert _collect_ids(output) == {"deep"}

    def test_empty_dataset(self) -> None:
        assert _collect_ids({"dataset": {}}) == set()

    def test_no_dataset_key(self) -> None:
        assert _collect_ids({}) == set()

    def test_multiple_levels(self) -> None:
        output: dict[str, Any] = {
            "dataset": {
                "files": [{"id": "root", "path": "root.png"}],
                "children": {
                    "child": {
                        "files": [{"id": "nested", "path": "nested.png"}]
                    }
                },
            }
        }
        assert _collect_ids(output) == {"root", "nested"}


# ===================================================================
# _collect_all_referenced_paths
# ===================================================================


class TestCollectAllReferencedPaths:
    def test_collects_file_paths(self) -> None:
        output: dict[str, Any] = {
            "dataset": {
                "files": [
                    {"path": "a.png", "id": "a"},
                    {"path": "b.png", "id": "b"},
                ]
            }
        }
        paths = _collect_all_referenced_paths(output)
        assert "a.png" in paths
        assert "b.png" in paths

    def test_collects_camera_paths(self) -> None:
        output: dict[str, Any] = {
            "dataset": {
                "children": {
                    "level1": {
                        "camera_intrinsics": "intrinsics.txt",
                        "camera_extrinsics": "extrinsics.txt",
                        "files": [{"path": "frame.png", "id": "f"}],
                    }
                }
            }
        }
        paths = _collect_all_referenced_paths(output)
        assert "intrinsics.txt" in paths
        assert "extrinsics.txt" in paths
        assert "frame.png" in paths

    def test_empty_dataset(self) -> None:
        assert _collect_all_referenced_paths({"dataset": {}}) == []

    def test_no_dataset_key(self) -> None:
        assert _collect_all_referenced_paths({}) == []


# ===================================================================
# Sampling
# ===================================================================


class TestSampling:
    def _make_flat_config(self, path: str) -> dict[str, Any]:
        return with_euler_train({
            "name": "test",
            "path": path,
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<name>.+)\.png$",
        })

    def test_sample_every_2nd(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(6):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg, sample=2)
        files = get_files(result)
        assert len(files) == 3

    def test_sample_every_3rd(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(9):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg, sample=3)
        files = get_files(result)
        assert len(files) == 3

    def test_sample_1_keeps_all(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result_sampled = index_dataset(cfg, sample=1)
        result_normal = index_dataset(cfg)
        assert get_files(result_sampled) == get_files(result_normal)

    def test_sample_none_keeps_all(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg, sample=None)
        files = get_files(result)
        assert len(files) == 4

    def test_sample_deterministic_ordering(self, tmp_path: Path) -> None:
        """Sorted order ensures deterministic sampling."""
        root = tmp_path / "data"
        for i in range(6):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result1 = index_dataset(cfg, sample=2)
        result2 = index_dataset(cfg, sample=2)
        assert get_files(result1) == get_files(result2)

    def test_sample_with_hierarchy(self, tmp_path: Path) -> None:
        """Sampling works globally across hierarchy levels."""
        root = tmp_path / "vkitti2"
        create_vkitti2_tree(root)
        cfg = make_vkitti2_config(str(root))
        full_result = index_dataset(cfg)
        sampled_result = index_dataset(cfg, sample=2)
        full_count = len(get_files(full_result))
        sampled_count = len(get_files(sampled_result))
        # Every 2nd file from 4 total files = 2
        assert sampled_count < full_count
        assert sampled_count == (full_count + 1) // 2

    def test_sample_via_from_files(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(6):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        ds_config = make_dataset_config(**cfg)

        from ds_crawler.handlers.generic import GenericHandler

        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        result = index_dataset_from_files(cfg, files, sample=2)
        assert len(get_files(result)) == 3

    def test_sampled_property_in_output(self, tmp_path: Path) -> None:
        """Output includes a 'sampled' key when sample is used."""
        root = tmp_path / "data"
        for i in range(6):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg, sample=3)
        assert result["sampled"] == 3

    def test_no_sampled_property_without_sampling(self, tmp_path: Path) -> None:
        """Output does NOT include 'sampled' when sample is None."""
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg)
        assert "sampled" not in result

    def test_sampled_property_via_from_files(self, tmp_path: Path) -> None:
        """index_dataset_from_files also sets 'sampled'."""
        root = tmp_path / "data"
        for i in range(6):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        ds_config = make_dataset_config(**cfg)

        from ds_crawler.handlers.generic import GenericHandler

        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        result = index_dataset_from_files(cfg, files, sample=2)
        assert result["sampled"] == 2


# ===================================================================
# Match index
# ===================================================================


class TestMatchIndex:
    def _make_flat_config(self, path: str) -> dict[str, Any]:
        return with_euler_train({
            "name": "test",
            "path": path,
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<name>.+)\.png$",
        })

    def test_match_index_filters_by_id(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))

        # Index all files
        full_result = index_dataset(cfg)
        full_files = get_files(full_result)
        assert len(full_files) == 4

        # Build a match index with only 2 IDs
        match = index_dataset(cfg, sample=2)
        match_ids = _collect_ids(match)
        assert len(match_ids) == 2

        # Re-index with match_index
        filtered = index_dataset(cfg, match_index=match)
        filtered_files = get_files(filtered)
        assert len(filtered_files) == 2

    def test_match_index_empty_produces_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        # Empty match index has no IDs
        empty_match: dict[str, Any] = {"dataset": {}}
        result = index_dataset(cfg, match_index=empty_match)
        assert get_files(result) == []

    def test_match_index_none_keeps_all(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        result = index_dataset(cfg, match_index=None)
        assert len(get_files(result)) == 4

    def test_match_index_cross_dataset(self, tmp_path: Path) -> None:
        """Use depth_predictions index to filter matching VKITTI2 entries."""
        # Create two datasets with overlapping IDs via scene names
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for name in ["001.png", "002.png", "003.png"]:
            touch(root_a / name)
            touch(root_b / name)

        cfg_a = self._make_flat_config(str(root_a))
        cfg_b = self._make_flat_config(str(root_b))

        # Index A with sample=2 (gets 001, 003)
        index_a = index_dataset(cfg_a, sample=2)
        # Use A's output as match_index for B
        result_b = index_dataset(cfg_b, match_index=index_a)
        files_b = get_files(result_b)
        assert len(files_b) == 2

    def test_match_index_and_sample_combined(self, tmp_path: Path) -> None:
        """match_index filters first, then sample subsamples the result."""
        root = tmp_path / "data"
        for i in range(10):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))

        # Full index has 10 files; sample=2 gives 5
        match = index_dataset(cfg, sample=2)
        assert len(get_files(match)) == 5

        # Now re-index: match_index filters to 5, then sample=2 from those -> ~3
        result = index_dataset(cfg, match_index=match, sample=2)
        result_files = get_files(result)
        # 5 IDs match, then every 2nd of those 5 -> 3
        assert len(result_files) == 3

    def test_match_index_via_from_files(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        for i in range(4):
            touch(root / f"frame_{i:03d}.png")
        cfg = self._make_flat_config(str(root))
        ds_config = make_dataset_config(**cfg)

        from ds_crawler.handlers.generic import GenericHandler

        handler = GenericHandler(ds_config)
        files = list(handler.get_files())

        match = index_dataset(cfg, sample=2)
        result = index_dataset_from_files(cfg, files, match_index=match)
        assert len(get_files(result)) == 2

    def test_match_index_hierarchy_qualified(self, tmp_path: Path) -> None:
        """match_index must respect hierarchy — same IDs under different levels
        should NOT cross-match.

        Reproduces a real-world bug where id_regex captures only the frame
        number (e.g. ``000001``) which repeats across scenes/cameras.  A flat
        set of IDs would collapse them and cause over-matching.
        """
        # Two datasets (e.g. rgb + segmentation) with same structure:
        #   scene_A/cam_0/frame_{0..3}.png
        #   scene_B/cam_0/frame_{0..3}.png
        # IDs are bare frame numbers → 4 unique, but 8 files total.
        root_rgb = tmp_path / "rgb"
        root_seg = tmp_path / "seg"
        for scene in ["scene_A", "scene_B"]:
            for frame in range(4):
                touch(root_rgb / scene / "cam_0" / f"{frame:03d}.png")
                touch(root_seg / scene / "cam_0" / f"{frame:03d}.png")

        cfg_base = with_euler_train({
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
            # id_regex captures only the bare frame number (NOT globally unique)
            "id_regex": r"^.+/(?P<frame>\d+)\.png$",
            "hierarchy_regex": r"^(?P<scene>[^/]+)/(?P<cam>[^/]+)/",
            "named_capture_group_value_separator": "_",
        })

        cfg_rgb = {**cfg_base, "name": "rgb", "path": str(root_rgb)}
        cfg_seg = {**cfg_base, "name": "seg", "path": str(root_seg)}

        # Index RGB with sample=2 → keeps 4 of 8 files
        rgb_index = index_dataset(cfg_rgb, sample=2)
        rgb_files = get_files(rgb_index)
        assert len(rgb_files) == 4

        # Only 4 unique bare frame IDs exist, so flat _collect_ids would
        # return at most 4.  But we have 4 *files*, spread across 2 scenes.
        # The qualified approach should give exactly those 4 (scene, cam, id)
        # tuples.

        # Index segmentation matched against RGB
        seg_index = index_dataset(cfg_seg, match_index=rgb_index)
        seg_files = get_files(seg_index)

        # Must match exactly the same 4 files — NOT all 8
        assert len(seg_files) == len(rgb_files)
