"""Tests for crawler.parser – helper functions and DatasetParser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from crawler.config import Config, DatasetConfig
from crawler.parser import (
    DatasetParser,
    _add_file_to_node,
    _deep_merge,
    _ensure_hierarchy_path,
    _get_hierarchy_keys,
)

from .conftest import (
    make_vkitti2_config,
    make_ddad_config,
    make_depth_predictions_config,
    touch,
    write_config_json,
)


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
        ds = DatasetConfig(
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
        ds = DatasetConfig(
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
        ds = DatasetConfig(
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


# ===================================================================
# DatasetParser._get_entry_hierarchy_keys
# ===================================================================


class TestGetEntryHierarchyKeys:
    def _make_parser(self, ds_config: DatasetConfig) -> DatasetParser:
        config = Config(datasets=[ds_config])
        return DatasetParser(config)

    def test_no_hierarchy_regex_returns_empty(self, tmp_path: Path) -> None:
        ds = DatasetConfig(
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
        ds = DatasetConfig(
            name="MyDS",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
            id_regex_join_char="+",
            properties={"type": "rgb", "gt": True},
        )
        parser = self._make_parser(ds)
        node: dict = {"children": {"a": {"files": []}}}
        output = parser._build_output(ds, node)

        assert output["name"] == "MyDS"
        assert output["id_regex"] == r"^(?P<f>.+)\.png$"
        assert output["id_regex_join_char"] == "+"
        assert output["type"] == "rgb"
        assert output["gt"] is True
        assert "dataset" in output
        assert output["dataset"]["children"]["a"]["files"] == []

    def test_hierarchy_regex_included(self, tmp_path: Path) -> None:
        ds = DatasetConfig(
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
        ds = DatasetConfig(
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

    def test_dataset_properties_deep_merged(self, tmp_path: Path) -> None:
        ds = DatasetConfig(
            name="test",
            path=str(tmp_path),
            type="rgb",
            basename_regex=r"^(?P<f>.+)\.png$",
            id_regex=r"^(?P<f>.+)\.png$",
            properties={
                "type": "rgb",
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
        ds_config = DatasetConfig(**cfg)
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
        ds_config = DatasetConfig(**cfg)
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
        ds_config = DatasetConfig(**cfg)
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

        ds = DatasetConfig(
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
        ds = DatasetConfig(
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
        ds = DatasetConfig(
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

        ds = DatasetConfig(
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

        ds = DatasetConfig(
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

        ds = DatasetConfig(
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

        ds = DatasetConfig(
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
    def test_strict_mode_raises_on_excessive_misses(
        self, tmp_path: Path
    ) -> None:
        """If >20% of files fail id_regex, strict mode raises."""
        root = tmp_path / "data"
        # Create 10 files, all with basenames that match but paths that don't
        for i in range(10):
            touch(root / f"wrong_dir_{i}" / "file.png")

        ds = DatasetConfig(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<f>.+)\.(?P<ext>png)$",
            # id_regex requires specific directory structure
            id_regex=r"^correct_dir/(?P<f>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config, strict=True)

        with pytest.raises(RuntimeError, match="failed ID extraction"):
            parser.parse_dataset(ds)

    def test_non_strict_does_not_raise(self, tmp_path: Path) -> None:
        """Non-strict mode warns but continues."""
        root = tmp_path / "data"
        for i in range(10):
            touch(root / f"wrong_dir_{i}" / "file.png")

        ds = DatasetConfig(
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
        ds_config = DatasetConfig(**cfg)
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
        ds_config = DatasetConfig(**cfg)
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
        ds_config = DatasetConfig(**cfg)
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

        ds = DatasetConfig(
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

        ds = DatasetConfig(
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
        assert paths[0] == root / "output.json"
        assert paths[0].exists()

    def test_write_per_dataset_custom_output_json(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.png")
        custom_output = tmp_path / "custom_output.json"

        ds = DatasetConfig(
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
