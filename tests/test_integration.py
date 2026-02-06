"""End-to-end integration tests and example output validation.

These tests verify that:
1. The shipped example_output.json has correct structure.
2. Parsing mock filesystems reproduces the expected output exactly.
3. The full pipeline (config → parse → JSON) works end-to-end.
4. The public API (index_dataset, index_dataset_from_path) works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.config import Config, DatasetConfig, CONFIG_FILENAME, load_dataset_config
from ds_crawler.parser import (
    DatasetParser,
    get_files,
    index_dataset,
    index_dataset_from_path,
)

from .conftest import (
    create_ddad_tree,
    create_depth_predictions_tree,
    create_vkitti2_tree,
    make_ddad_config,
    make_depth_predictions_config,
    make_vkitti2_config,
    touch,
    write_config_json,
)


# ===================================================================
# Helpers
# ===================================================================


def collect_all_files(node: dict) -> list[dict]:
    """Recursively collect all file entries from a hierarchy node."""
    files = list(node.get("files", []))
    for child in node.get("children", {}).values():
        files.extend(collect_all_files(child))
    return files


def get_node_at_path(root: dict, keys: list[str]) -> dict:
    """Navigate a hierarchy by a list of children keys."""
    current = root
    for key in keys:
        current = current["children"][key]
    return current


def with_euler_train(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure a config dict has minimal required euler_train metadata."""
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
        result["properties"] = props
    return result


def make_dataset_config(**kwargs: Any) -> DatasetConfig:
    """Build DatasetConfig with required euler_train defaults when omitted."""
    return DatasetConfig(**with_euler_train(kwargs))


# ===================================================================
# Example output.json structural validation
# ===================================================================


class TestExampleOutputStructure:
    """Validate the shipped example_output.json is well-formed."""

    def test_is_list_of_datasets(self, example_output_data: list[dict]) -> None:
        assert isinstance(example_output_data, list)
        assert len(example_output_data) == 3

    def test_each_dataset_has_required_keys(
        self, example_output_data: list[dict]
    ) -> None:
        for ds in example_output_data:
            assert "name" in ds
            assert "type" in ds
            assert "id_regex" in ds
            assert "id_regex_join_char" in ds
            assert "euler_train" in ds
            assert "dataset" in ds

    def test_dataset_names(self, example_output_data: list[dict]) -> None:
        names = [ds["name"] for ds in example_output_data]
        assert names == ["VKITTI2", "DDAD", "depth_predictions"]

    def test_vkitti2_top_level_properties(
        self, example_output_data: list[dict]
    ) -> None:
        vk = example_output_data[0]
        assert vk["type"] == "rgb"
        assert vk["gt"] is True
        assert vk["baseline"] is True
        assert vk["euler_train"]["used_as"] == "target"
        assert vk["euler_train"]["slot"] == "demo.target.rgb"
        assert vk["euler_train"]["modality_type"] == "rgb"
        assert vk["id_regex_join_char"] == "+"
        assert "hierarchy_regex" in vk
        assert vk["named_capture_group_value_separator"] == ":"

    def test_vkitti2_dataset_metadata(
        self, example_output_data: list[dict]
    ) -> None:
        vk_ds = example_output_data[0]["dataset"]
        assert vk_ds["license"] == "CC BY-NC-SA 4.0"
        assert "source" in vk_ds

    def test_vkitti2_hierarchy_depth(
        self, example_output_data: list[dict]
    ) -> None:
        """VKITTI2 should have scene → variation → camera → frame hierarchy."""
        vk_ds = example_output_data[0]["dataset"]
        assert "children" in vk_ds

        # Scene01 exists
        assert "scene:Scene01" in vk_ds["children"]
        scene01 = vk_ds["children"]["scene:Scene01"]

        # variation:clone exists
        assert "variation:clone" in scene01["children"]
        clone = scene01["children"]["variation:clone"]

        # camera:Camera_0 exists
        assert "camera:Camera_0" in clone["children"]
        cam0 = clone["children"]["camera:Camera_0"]

        # frame:00001 exists with files
        assert "frame:00001" in cam0["children"]
        frame = cam0["children"]["frame:00001"]
        assert "files" in frame
        assert len(frame["files"]) == 1

    def test_vkitti2_file_entry_structure(
        self, example_output_data: list[dict]
    ) -> None:
        """Each file entry should have path, id, path_properties, basename_properties."""
        vk_ds = example_output_data[0]["dataset"]
        frame = get_node_at_path(
            vk_ds,
            ["scene:Scene01", "variation:clone", "camera:Camera_0", "frame:00001"],
        )
        entry = frame["files"][0]

        assert entry["path"] == "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"
        assert entry["id"] == "scene-Scene01+variation-clone+camera-Camera_0+frame-00001"
        assert entry["path_properties"] == {
            "scene": "Scene01",
            "variation": "clone",
            "camera": "Camera_0",
        }
        assert entry["basename_properties"] == {"frame": "00001", "ext": "jpg"}

    def test_vkitti2_camera_files(
        self, example_output_data: list[dict]
    ) -> None:
        vk_ds = example_output_data[0]["dataset"]
        cam0 = get_node_at_path(
            vk_ds,
            ["scene:Scene01", "variation:clone", "camera:Camera_0"],
        )
        assert cam0["camera_intrinsics"] == "Scene01/clone/intrinsics/Camera_0_intrinsics.txt"
        assert cam0["camera_extrinsics"] == "Scene01/clone/extrinsics/Camera_0_extrinsics.txt"

    def test_vkitti2_scene02(
        self, example_output_data: list[dict]
    ) -> None:
        vk_ds = example_output_data[0]["dataset"]
        assert "scene:Scene02" in vk_ds["children"]
        fog_cam0 = get_node_at_path(
            vk_ds, ["scene:Scene02", "variation:fog", "camera:Camera_0"]
        )
        assert fog_cam0["camera_intrinsics"] == "Scene02/fog/intrinsics/Camera_0_intrinsics.txt"
        assert fog_cam0["camera_extrinsics"] == "Scene02/fog/extrinsics/Camera_0_extrinsics.txt"

    def test_ddad_structure(self, example_output_data: list[dict]) -> None:
        ddad = example_output_data[1]
        assert ddad["type"] == "rgb"
        assert ddad["gt"] is True
        assert ddad["euler_train"]["used_as"] == "target"

        ddad_ds = ddad["dataset"]
        cam01 = get_node_at_path(
            ddad_ds, ["scene:000150", "camera:CAMERA_01"]
        )
        assert cam01["camera_intrinsics"] == "000150/calibration/CAMERA_01.json"

        # Two frames
        assert "frame:0000000050" in cam01["children"]
        assert "frame:0000000051" in cam01["children"]

    def test_ddad_file_ids(self, example_output_data: list[dict]) -> None:
        ddad_ds = example_output_data[1]["dataset"]
        frame = get_node_at_path(
            ddad_ds,
            ["scene:000150", "camera:CAMERA_01", "frame:0000000050"],
        )
        entry = frame["files"][0]
        assert entry["id"] == "scene-000150+camera-CAMERA_01+frame-0000000050"

    def test_depth_predictions_structure(
        self, example_output_data: list[dict]
    ) -> None:
        dp = example_output_data[2]
        assert dp["type"] == "depth"
        assert dp["gt"] is False
        assert dp["model"] == "DepthAnythingV2"
        assert dp["euler_train"]["used_as"] == "input"
        assert dp["euler_train"]["slot"] == "demo.input.depth"

        dp_ds = dp["dataset"]
        frame01 = get_node_at_path(dp_ds, ["scene:Scene01", "frame:00001"])
        assert len(frame01["files"]) == 2

    def test_depth_predictions_dual_extensions(
        self, example_output_data: list[dict]
    ) -> None:
        dp_ds = example_output_data[2]["dataset"]
        frame01 = get_node_at_path(dp_ds, ["scene:Scene01", "frame:00001"])
        exts = {f["basename_properties"]["ext"] for f in frame01["files"]}
        assert exts == {"png", "npy"}

        ids = {f["id"] for f in frame01["files"]}
        assert "scene-Scene01+frame-00001+ext-png" in ids
        assert "scene-Scene01+frame-00001+ext-npy" in ids

    def test_all_file_ids_unique_per_dataset(
        self, example_output_data: list[dict]
    ) -> None:
        """Every file ID within a dataset should be unique."""
        for ds in example_output_data:
            files = collect_all_files(ds["dataset"])
            ids = [f["id"] for f in files]
            assert len(ids) == len(set(ids)), (
                f"Duplicate IDs in dataset '{ds['name']}': "
                f"{[x for x in ids if ids.count(x) > 1]}"
            )

    def test_all_file_paths_are_relative(
        self, example_output_data: list[dict]
    ) -> None:
        """File paths should be relative (no leading /)."""
        for ds in example_output_data:
            files = collect_all_files(ds["dataset"])
            for f in files:
                assert not f["path"].startswith("/"), (
                    f"Path should be relative: {f['path']}"
                )


# ===================================================================
# End-to-end: reproduce example output from mock filesystem
# ===================================================================


class TestReproduceExampleOutput:
    """Create mock filesystem matching the example and verify parsed output
    matches the expected example_output.json."""

    def _strip_ordering(self, obj: Any) -> Any:
        """Normalize file lists by sorting them by path for stable comparison."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == "files" and isinstance(value, list):
                    result[key] = sorted(
                        [self._strip_ordering(v) for v in value],
                        key=lambda f: f["path"],
                    )
                else:
                    result[key] = self._strip_ordering(value)
            return result
        if isinstance(obj, list):
            return [self._strip_ordering(item) for item in obj]
        return obj

    def test_vkitti2_matches_example(
        self, vkitti2_root: Path, example_output_data: list[dict]
    ) -> None:
        cfg = make_vkitti2_config(str(vkitti2_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)
        results = parser.parse_all()

        actual = self._strip_ordering(results[0])
        expected = self._strip_ordering(example_output_data[0])

        # Compare hierarchical structure
        assert actual["name"] == expected["name"]
        assert actual["id_regex_join_char"] == expected["id_regex_join_char"]
        assert actual["type"] == expected["type"]
        assert actual["gt"] == expected["gt"]
        assert actual["baseline"] == expected["baseline"]
        assert actual["hierarchy_regex"] == expected["hierarchy_regex"]
        assert (
            actual["named_capture_group_value_separator"]
            == expected["named_capture_group_value_separator"]
        )

        # Compare the dataset hierarchy
        actual_ds = actual["dataset"]
        expected_ds = expected["dataset"]

        # Check metadata properties are merged
        assert actual_ds["license"] == expected_ds["license"]
        assert actual_ds["source"] == expected_ds["source"]

        # Compare children structure
        assert set(actual_ds["children"].keys()) == set(expected_ds["children"].keys())

        # Deep comparison of Scene01/clone/Camera_0
        actual_cam0 = get_node_at_path(
            actual_ds,
            ["scene:Scene01", "variation:clone", "camera:Camera_0"],
        )
        expected_cam0 = get_node_at_path(
            expected_ds,
            ["scene:Scene01", "variation:clone", "camera:Camera_0"],
        )
        assert actual_cam0["camera_intrinsics"] == expected_cam0["camera_intrinsics"]
        assert actual_cam0["camera_extrinsics"] == expected_cam0["camera_extrinsics"]

        # Compare frame files
        for frame_key in ["frame:00001", "frame:00002"]:
            if frame_key in expected_cam0.get("children", {}):
                actual_frame = actual_cam0["children"][frame_key]
                expected_frame = expected_cam0["children"][frame_key]
                assert actual_frame["files"] == expected_frame["files"], (
                    f"Mismatch in {frame_key}"
                )

    def test_ddad_matches_example(
        self, ddad_root: Path, example_output_data: list[dict]
    ) -> None:
        cfg = make_ddad_config(str(ddad_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)
        results = parser.parse_all()

        actual = self._strip_ordering(results[0])
        expected = self._strip_ordering(example_output_data[1])

        assert actual["name"] == expected["name"]
        assert actual["type"] == expected["type"]

        # DDAD Camera_01 intrinsics
        actual_cam = get_node_at_path(
            actual["dataset"],
            ["scene:000150", "camera:CAMERA_01"],
        )
        expected_cam = get_node_at_path(
            expected["dataset"],
            ["scene:000150", "camera:CAMERA_01"],
        )
        assert actual_cam["camera_intrinsics"] == expected_cam["camera_intrinsics"]

        # Compare frame files
        for frame_key in ["frame:0000000050", "frame:0000000051"]:
            actual_frame = actual_cam["children"][frame_key]
            expected_frame = expected_cam["children"][frame_key]
            assert actual_frame["files"] == expected_frame["files"]

    def test_depth_predictions_matches_example(
        self, depth_predictions_root: Path, example_output_data: list[dict]
    ) -> None:
        cfg = make_depth_predictions_config(str(depth_predictions_root))
        ds_config = make_dataset_config(**cfg)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)
        results = parser.parse_all()

        actual = self._strip_ordering(results[0])
        expected = self._strip_ordering(example_output_data[2])

        assert actual["name"] == expected["name"]
        assert actual["type"] == expected["type"]
        assert actual["gt"] == expected["gt"]
        assert actual["model"] == expected["model"]

        # Compare frame:00001 files (should have .png and .npy)
        actual_frame = get_node_at_path(
            actual["dataset"],
            ["scene:Scene01", "frame:00001"],
        )
        expected_frame = get_node_at_path(
            expected["dataset"],
            ["scene:Scene01", "frame:00001"],
        )
        assert actual_frame["files"] == expected_frame["files"]


# ===================================================================
# Full pipeline: config file → parse → JSON output
# ===================================================================


class TestFullPipeline:
    def test_config_to_json_roundtrip(self, tmp_path: Path) -> None:
        """Config → mock filesystem → parse → write JSON → read JSON."""
        data_root = tmp_path / "datasets"

        vk_root = data_root / "vkitti2"
        create_vkitti2_tree(vk_root)
        ddad_root = data_root / "ddad"
        create_ddad_tree(ddad_root)
        dp_root = data_root / "depth_preds"
        create_depth_predictions_tree(dp_root)

        datasets = [
            make_vkitti2_config(str(vk_root)),
            make_ddad_config(str(ddad_root)),
            make_depth_predictions_config(str(dp_root)),
        ]
        config_path = write_config_json(tmp_path / "config.json", datasets)

        config = Config.from_file(config_path)
        parser = DatasetParser(config)

        output_path = tmp_path / "output.json"
        parser.write_output(output_path)

        with open(output_path) as f:
            output = json.load(f)

        assert isinstance(output, list)
        assert len(output) == 3
        assert output[0]["name"] == "VKITTI2"
        assert output[1]["name"] == "DDAD"
        assert output[2]["name"] == "depth_predictions"

        # Verify files are present
        for ds_output in output:
            files = collect_all_files(ds_output["dataset"])
            assert len(files) > 0, f"No files in dataset '{ds_output['name']}'"

    def test_per_dataset_output(self, tmp_path: Path) -> None:
        """Each dataset writes its own output JSON."""
        root = tmp_path / "data"
        create_depth_predictions_tree(root)

        cfg_data = make_depth_predictions_config(str(root))
        config_path = write_config_json(tmp_path / "config.json", [cfg_data])
        config = Config.from_file(config_path)
        parser = DatasetParser(config)

        paths = parser.write_outputs_per_dataset()
        assert len(paths) == 1
        assert paths[0].exists()

        with open(paths[0]) as f:
            ds_output = json.load(f)
        assert ds_output["name"] == "depth_predictions"

    def test_strict_mode_pipeline(self, tmp_path: Path) -> None:
        """Strict mode should work fine when there are no issues."""
        root = tmp_path / "data"
        create_depth_predictions_tree(root)

        cfg_data = make_depth_predictions_config(str(root))
        config_path = write_config_json(tmp_path / "config.json", [cfg_data])
        config = Config.from_file(config_path)
        parser = DatasetParser(config, strict=True)

        results = parser.parse_all()
        assert len(results) == 1

        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 4  # 2 frames × 2 extensions

    def test_workdir_pipeline(self, tmp_path: Path) -> None:
        """Test workdir parameter in end-to-end pipeline."""
        workdir = tmp_path / "work"
        data_root = workdir / "rel_data"
        create_depth_predictions_tree(data_root)

        cfg_data = {
            "name": "test",
            "path": "rel_data",
            "type": "depth",
            "file_extensions": [".png", ".npy"],
            "basename_regex": r"^(?P<frame>\d+)_pred\.(?P<ext>png|npy)$",
            "id_regex": r"^(?P<scene>[^/]+)/(?P<frame>\d+)_pred\.(?P<ext>png|npy)$",
            "hierarchy_regex": r"^(?P<scene>[^/]+)/(?P<frame>\d+)_pred\.(?:png|npy)$",
            "named_capture_group_value_separator": ":",
            "properties": {
                "euler_train": {
                    "used_as": "input",
                    "modality_type": "depth",
                }
            },
        }
        config_path = write_config_json(tmp_path / "config.json", [cfg_data])
        config = Config.from_file(config_path, workdir=str(workdir))
        parser = DatasetParser(config)
        results = parser.parse_all()

        assert len(results) == 1
        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 4


# ===================================================================
# Edge cases for the full pipeline
# ===================================================================


class TestEdgeCases:
    def test_single_file_dataset(self, tmp_path: Path) -> None:
        root = tmp_path / "single"
        touch(root / "only_file.png")

        ds = make_dataset_config(
            name="single",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<name>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<name>.+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 1
        assert files[0]["id"] == "name-only_file"

    def test_deeply_nested_hierarchy(self, tmp_path: Path) -> None:
        root = tmp_path / "deep"
        touch(root / "a" / "b" / "c" / "001.png")

        ds = make_dataset_config(
            name="deep",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
            id_regex=r"^(?P<l1>[^/]+)/(?P<l2>[^/]+)/(?P<l3>[^/]+)/(?P<frame>\d+)\.png$",
            hierarchy_regex=r"^(?P<l1>[^/]+)/(?P<l2>[^/]+)/(?P<l3>[^/]+)/(?P<frame>\d+)\.png$",
            named_capture_group_value_separator=":",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        ds_data = results[0]["dataset"]
        node = get_node_at_path(
            ds_data, ["l1:a", "l2:b", "l3:c", "frame:001"]
        )
        assert len(node["files"]) == 1

    def test_multiple_files_same_hierarchy_level(self, tmp_path: Path) -> None:
        """Multiple files at the same hierarchy level with different IDs."""
        root = tmp_path / "multi"
        touch(root / "scene" / "001.png")
        touch(root / "scene" / "002.png")
        touch(root / "scene" / "003.png")

        ds = make_dataset_config(
            name="multi",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
            id_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
            hierarchy_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
            named_capture_group_value_separator=":",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 3
        ids = {f["id"] for f in files}
        assert len(ids) == 3

    def test_mixed_matching_and_nonmatching_files(self, tmp_path: Path) -> None:
        """Some files match regexes, some don't – only matching ones in output."""
        root = tmp_path / "mixed"
        touch(root / "valid" / "001.png")
        touch(root / "valid" / "002.png")
        touch(root / "stray_file.png")  # basename matches but id_regex won't

        ds = make_dataset_config(
            name="mixed",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
            id_regex=r"^valid/(?P<frame>\d+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        # stray_file.png fails basename_regex (name doesn't match \d+),
        # so it's skipped
        assert len(files) == 2

    def test_custom_id_join_char(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a" / "001.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
            id_regex=r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
            id_regex_join_char="/",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        assert files[0]["id"] == "dir-a/frame-001"

    def test_no_path_regex_gives_empty_path_properties(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "data"
        touch(root / "001.png")

        ds = make_dataset_config(
            name="test",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<frame>\d+)\.(?P<ext>png)$",
            id_regex=r"^(?P<frame>\d+)\.png$",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        assert files[0]["path_properties"] == {}

    def test_large_dataset_no_crash(self, tmp_path: Path) -> None:
        """Stress test with many files – parser shouldn't crash."""
        root = tmp_path / "large"
        for i in range(200):
            touch(root / f"scene_{i:03d}" / "frame.png")

        ds = make_dataset_config(
            name="large",
            path=str(root),
            type="rgb",
            file_extensions=[".png"],
            basename_regex=r"^(?P<name>.+)\.(?P<ext>png)$",
            id_regex=r"^(?P<scene>[^/]+)/(?P<name>.+)\.png$",
            hierarchy_regex=r"^(?P<scene>[^/]+)/(?P<name>.+)\.png$",
            named_capture_group_value_separator=":",
        )
        config = Config(datasets=[ds])
        parser = DatasetParser(config)
        results = parser.parse_all()

        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 200


# ===================================================================
# Public API: index_dataset
# ===================================================================


class TestIndexDataset:
    """Test the index_dataset() convenience function."""

    def test_returns_output_dict(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "scene" / "001.png")

        config_dict = with_euler_train({
            "name": "test",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
        })
        result = index_dataset(config_dict)

        assert result["name"] == "test"
        assert result["id_regex"] == config_dict["id_regex"]
        assert "dataset" in result
        files = collect_all_files(result["dataset"])
        assert len(files) == 1
        assert files[0]["id"] == "dir-scene+frame-001"

    def test_with_hierarchy(self, vkitti2_root: Path) -> None:
        config_dict = make_vkitti2_config(str(vkitti2_root))
        result = index_dataset(config_dict)

        assert result["name"] == "VKITTI2"
        assert result["type"] == "rgb"
        assert "hierarchy_regex" in result
        files = collect_all_files(result["dataset"])
        assert len(files) > 0

    def test_strict_mode(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a" / "001.png")
        touch(root / "b" / "001.png")

        config_dict = with_euler_train({
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
            index_dataset(config_dict, strict=True)


# ===================================================================
# Public API: index_dataset_from_path with ds-crawler.config
# ===================================================================


class TestIndexDatasetFromPath:
    """Test the index_dataset_from_path() function and ds-crawler.config loading."""

    def _write_config_file(self, dataset_root: Path, config: dict) -> Path:
        config_path = dataset_root / CONFIG_FILENAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(with_euler_train(config), f)
        return config_path

    def test_basic_from_path(self, tmp_path: Path) -> None:
        root = tmp_path / "my_dataset"
        touch(root / "scene" / "001.png")

        self._write_config_file(root, {
            "name": "from_path_test",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
        })

        result = index_dataset_from_path(root)
        assert result["name"] == "from_path_test"
        files = collect_all_files(result["dataset"])
        assert len(files) == 1

    def test_config_file_path_overridden(self, tmp_path: Path) -> None:
        """The path passed to the function overrides any path in ds-crawler.config."""
        root = tmp_path / "actual_data"
        touch(root / "001.png")

        # Config file has a different path, but caller's path should win
        self._write_config_file(root, {
            "name": "override_test",
            "path": "/some/other/path",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        })

        result = index_dataset_from_path(root)
        assert result["name"] == "override_test"
        files = collect_all_files(result["dataset"])
        assert len(files) == 1

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "no_config"
        root.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=CONFIG_FILENAME):
            index_dataset_from_path(root)

    def test_config_from_file_path_only_entry(self, tmp_path: Path) -> None:
        """Config.from_file with a path-only dataset entry resolves ds-crawler.config."""
        root = tmp_path / "ds_root"
        touch(root / "file.png")

        self._write_config_file(root, {
            "name": "cfg_file_test",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<f>.+)\.png$",
        })

        # Config JSON with only path
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"datasets": [{"path": str(root)}]}, f)

        config = Config.from_file(config_path)
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "cfg_file_test"

        parser = DatasetParser(config)
        results = parser.parse_all()
        files = collect_all_files(results[0]["dataset"])
        assert len(files) == 1

    def test_load_dataset_config_full_inline(self, tmp_path: Path) -> None:
        """load_dataset_config with all fields present does not look for config file."""
        config_dict = with_euler_train({
            "name": "inline",
            "path": str(tmp_path),
            "type": "rgb",
            "basename_regex": r"^(?P<f>.+)\.png$",
            "id_regex": r"^(?P<f>.+)\.png$",
        })
        ds_config = load_dataset_config(config_dict)
        assert ds_config.name == "inline"

    def test_path_only_with_workdir(self, tmp_path: Path) -> None:
        """Path-only entry combined with workdir resolves correctly."""
        workdir = tmp_path / "work"
        root = workdir / "rel_ds"
        touch(root / "frame.png")

        self._write_config_file(root, {
            "name": "workdir_test",
            "path": "rel_ds",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<f>.+)\.png$",
        })

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"datasets": [{"path": "rel_ds"}]}, f)

        config = Config.from_file(config_path, workdir=str(workdir))
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "workdir_test"


# ===================================================================
# save_index option
# ===================================================================


class TestSaveIndex:
    """Test the save_index parameter on index_dataset / index_dataset_from_path."""

    def test_index_dataset_save_index(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "001.png")

        config_dict = with_euler_train({
            "name": "save_test",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        })
        result = index_dataset(config_dict, save_index=True)

        output_path = root / ".ds_crawler" / "output.json"
        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["name"] == result["name"]
        assert saved["dataset"] == result["dataset"]

    def test_index_dataset_no_save_by_default(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "001.png")

        index_dataset(with_euler_train({
            "name": "no_save",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }))
        assert not (root / ".ds_crawler" / "output.json").exists()

    def test_index_dataset_from_path_save_index(self, tmp_path: Path) -> None:
        root = tmp_path / "ds"
        touch(root / "001.png")

        config_file = root / CONFIG_FILENAME
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(with_euler_train({
                "name": "save_from_path",
                "path": str(root),
                "type": "rgb",
                "file_extensions": [".png"],
                "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
                "id_regex": r"^(?P<frame>\d+)\.png$",
            }), f)

        result = index_dataset_from_path(root, save_index=True)

        output_path = root / ".ds_crawler" / "output.json"
        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["name"] == result["name"]


# ===================================================================
# Cache bypass with sample / match_index
# ===================================================================


class TestFromPathCacheBypass:
    """index_dataset_from_path must re-compute when sample or match_index is set."""

    def _setup_dataset(self, tmp_path: Path, n_files: int = 6) -> Path:
        root = tmp_path / "ds"
        for i in range(n_files):
            touch(root / f"frame_{i:03d}.png")

        config_path = root / CONFIG_FILENAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(with_euler_train({
                "name": "cache_bypass_test",
                "path": str(root),
                "type": "rgb",
                "file_extensions": [".png"],
                "basename_regex": r"^(?P<f>.+)\.(?P<ext>png)$",
                "id_regex": r"^(?P<name>.+)\.png$",
            }), f)

        return root

    def test_sample_bypasses_cache(self, tmp_path: Path) -> None:
        root = self._setup_dataset(tmp_path, n_files=6)

        # First call without sampling, saving index
        full = index_dataset_from_path(root, save_index=True)
        assert (root / ".ds_crawler" / "output.json").exists()
        assert len(get_files(full)) == 6

        # Second call with sample — must NOT return the cached full index
        sampled = index_dataset_from_path(root, sample=2)
        assert len(get_files(sampled)) == 3
        assert sampled["sampled"] == 2

    def test_match_index_bypasses_cache(self, tmp_path: Path) -> None:
        root = self._setup_dataset(tmp_path, n_files=6)

        # First call: index and save
        full = index_dataset_from_path(root, save_index=True)
        assert len(get_files(full)) == 6

        # Build a match index with only 3 IDs
        match = index_dataset_from_path(root, sample=2)
        assert len(get_files(match)) == 3

        # Re-index with match_index — must NOT return the cached full index
        filtered = index_dataset_from_path(root, match_index=match)
        assert len(get_files(filtered)) == 3

    def test_sample_and_match_index_bypass_cache(self, tmp_path: Path) -> None:
        root = self._setup_dataset(tmp_path, n_files=10)

        full = index_dataset_from_path(root, save_index=True)
        assert len(get_files(full)) == 10

        # match_index with 5 IDs, then sample every 2nd → 3
        match = index_dataset_from_path(root, sample=2)
        result = index_dataset_from_path(root, match_index=match, sample=2)
        assert len(get_files(result)) == 3
        assert result["sampled"] == 2

    def test_no_sample_still_uses_cache(self, tmp_path: Path) -> None:
        """Without sample or match_index, caching still works normally."""
        root = self._setup_dataset(tmp_path, n_files=4)

        index_dataset_from_path(root, save_index=True)
        # Plant a stale marker to confirm cache is read
        output_path = root / ".ds_crawler" / "output.json"
        with open(output_path) as f:
            cached = json.load(f)
        cached["_stale_marker"] = True
        with open(output_path, "w") as f:
            json.dump(cached, f)

        second = index_dataset_from_path(root)
        assert second.get("_stale_marker") is True

    def test_sampled_property_persisted_in_saved_index(self, tmp_path: Path) -> None:
        """When save_index is True and sample is set, output.json has 'sampled'."""
        root = self._setup_dataset(tmp_path, n_files=6)

        index_dataset_from_path(root, sample=3, save_index=True)

        with open(root / ".ds_crawler" / "output.json") as f:
            saved = json.load(f)
        assert saved["sampled"] == 3
