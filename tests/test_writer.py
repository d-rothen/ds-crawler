"""Tests for DatasetWriter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler import DatasetWriter, align_datasets, index_dataset
from .conftest import touch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EULER_TRAIN = {"used_as": "target", "modality_type": "semantic"}


def _make_writer(tmp_path: Path, **overrides: Any) -> DatasetWriter:
    defaults: dict[str, Any] = {
        "root": tmp_path / "output",
        "name": "segmentation",
        "type": "segmentation",
        "euler_train": _EULER_TRAIN,
        "separator": ":",
    }
    defaults.update(overrides)
    return DatasetWriter(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        assert len(writer) == 0
        assert writer.root == tmp_path / "output"

    def test_missing_used_as_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="used_as"):
            _make_writer(tmp_path, euler_train={"modality_type": "depth"})

    def test_missing_modality_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="modality_type"):
            _make_writer(tmp_path, euler_train={"used_as": "target"})

    def test_depth_modality_requires_meta(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta is required"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "depth"},
            )

    def test_depth_modality_meta_missing_field(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta.scale_to_meters is required"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "depth"},
                meta={"radial_depth": False, "range": [0, 65535]},
            )

    def test_depth_modality_meta_wrong_type(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta.radial_depth must be a bool"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "depth"},
                meta={"radial_depth": "yes", "scale_to_meters": 1.0, "range": [0, 65535]},
            )

    def test_depth_modality_meta_valid(self, tmp_path: Path) -> None:
        writer = _make_writer(
            tmp_path,
            euler_train={"used_as": "target", "modality_type": "depth"},
            meta={"radial_depth": False, "scale_to_meters": 0.001, "range": [0, 65535]},
        )
        assert len(writer) == 0

    def test_non_depth_modality_no_meta_required(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)  # uses "semantic" modality
        assert len(writer) == 0

    def test_semseg_modality_requires_meta(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta is required"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "semantic_segmentation"},
            )

    def test_semseg_skyclass_invalid(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta.skyclass must be"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "semantic_segmentation"},
                meta={"skyclass": [0, 128]},
            )

    def test_semseg_skyclass_valid(self, tmp_path: Path) -> None:
        writer = _make_writer(
            tmp_path,
            euler_train={"used_as": "target", "modality_type": "semantic_segmentation"},
            meta={"skyclass": [135, 206, 235]},
        )
        assert len(writer) == 0

    def test_rgb_modality_requires_meta(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta is required"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "rgb"},
            )

    def test_rgb_range_invalid(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="meta.range must be"):
            _make_writer(
                tmp_path,
                euler_train={"used_as": "target", "modality_type": "rgb"},
                meta={"range": [255, 0]},
            )

    def test_rgb_range_valid(self, tmp_path: Path) -> None:
        writer = _make_writer(
            tmp_path,
            euler_train={"used_as": "target", "modality_type": "rgb"},
            meta={"range": [0, 255]},
        )
        assert len(writer) == 0


# ---------------------------------------------------------------------------
# get_path — hierarchy + directory creation
# ---------------------------------------------------------------------------


class TestGetPath:
    def test_creates_directories_and_returns_abs_path(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        p = writer.get_path("/scene:Scene01/camera:Cam0/00001", "00001.png")

        assert p == tmp_path / "output" / "Scene01" / "Cam0" / "00001.png"
        assert p.parent.is_dir()

    def test_leading_slash_optional(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        p1 = writer.get_path("/scene:S1/00001", "00001.png")
        # Same without leading slash — should produce identical path
        writer2 = _make_writer(tmp_path, root=tmp_path / "output2")
        p2 = writer2.get_path("scene:S1/00001", "00001.png")

        assert p1.relative_to(tmp_path / "output") == p2.relative_to(tmp_path / "output2")

    def test_flat_dataset_no_hierarchy(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        p = writer.get_path("/00001", "00001.png")

        assert p == tmp_path / "output" / "00001.png"
        assert p.parent.is_dir()

    def test_unnamed_hierarchy_keys(self, tmp_path: Path) -> None:
        """Keys without separator are used as-is for directories."""
        writer = _make_writer(tmp_path, separator=None)
        p = writer.get_path("/Scene01/clone/Camera_0/00001", "00001.png")

        assert p == tmp_path / "output" / "Scene01" / "clone" / "Camera_0" / "00001.png"

    def test_increments_count(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        assert len(writer) == 0
        writer.get_path("/scene:S1/001", "001.png")
        assert len(writer) == 1
        writer.get_path("/scene:S1/002", "002.png")
        assert len(writer) == 2

    def test_empty_full_id_raises(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            writer.get_path("/", "file.png")

    def test_source_meta_overrides_properties(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        source_meta = {
            "path_properties": {"scene": "Scene01", "variation": "clone"},
            "basename_properties": {"frame": "00001", "ext": "jpg"},
        }
        writer.get_path("/scene:Scene01/variation:clone/00001", "00001.png", source_meta=source_meta)

        output = writer.build_output()
        entry = output["dataset"]["children"]["scene:Scene01"]["children"]["variation:clone"]["files"][0]
        assert entry["path_properties"] == {"scene": "Scene01", "variation": "clone"}
        assert entry["basename_properties"] == {"frame": "00001", "ext": "jpg"}


# ---------------------------------------------------------------------------
# build_output — structure
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_single_entry(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/scene:Scene01/camera:Cam0/00001", "00001.png")

        output = writer.build_output()

        assert output["name"] == "segmentation"
        assert output["type"] == "segmentation"
        assert output["euler_train"] == _EULER_TRAIN
        assert output["named_capture_group_value_separator"] == ":"

        # Navigate hierarchy
        scene_node = output["dataset"]["children"]["scene:Scene01"]
        cam_node = scene_node["children"]["camera:Cam0"]
        assert len(cam_node["files"]) == 1

        entry = cam_node["files"][0]
        assert entry["id"] == "00001"
        assert entry["path"] == "Scene01/Cam0/00001.png"
        assert entry["path_properties"] == {"scene": "Scene01", "camera": "Cam0"}
        assert entry["basename_properties"] == {"ext": "png"}

    def test_multiple_entries_same_hierarchy(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/scene:S1/cam:C0/001", "001.png")
        writer.get_path("/scene:S1/cam:C0/002", "002.png")

        output = writer.build_output()
        files = output["dataset"]["children"]["scene:S1"]["children"]["cam:C0"]["files"]
        assert len(files) == 2
        assert {f["id"] for f in files} == {"001", "002"}

    def test_multiple_hierarchy_branches(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/scene:S1/001", "001.png")
        writer.get_path("/scene:S2/001", "001.png")

        output = writer.build_output()
        children = output["dataset"]["children"]
        assert set(children.keys()) == {"scene:S1", "scene:S2"}
        assert len(children["scene:S1"]["files"]) == 1
        assert len(children["scene:S2"]["files"]) == 1

    def test_no_separator_omits_key(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path, separator=None)
        output = writer.build_output()
        assert "named_capture_group_value_separator" not in output

    def test_extra_properties_included(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path, gt=False, model="MyModel")
        output = writer.build_output()
        assert output["gt"] is False
        assert output["model"] == "MyModel"

    def test_flat_dataset(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/001", "001.png")
        writer.get_path("/002", "002.png")

        output = writer.build_output()
        files = output["dataset"]["files"]
        assert len(files) == 2
        assert files[0]["path"] == "001.png"
        assert files[0]["path_properties"] == {}


# ---------------------------------------------------------------------------
# save_index — persistence
# ---------------------------------------------------------------------------


class TestSaveIndex:
    def test_writes_to_ds_crawler_dir(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/scene:S1/001", "001.png")

        result_path = writer.save_index()

        assert result_path == tmp_path / "output" / ".ds_crawler" / "output.json"
        assert result_path.is_file()

        with open(result_path) as f:
            data = json.load(f)
        assert data["name"] == "segmentation"
        assert len(data["dataset"]["children"]["scene:S1"]["files"]) == 1

    def test_custom_filename(self, tmp_path: Path) -> None:
        writer = _make_writer(tmp_path)
        writer.get_path("/001", "001.png")

        result_path = writer.save_index(filename="custom.json")

        assert result_path == tmp_path / "output" / ".ds_crawler" / "custom.json"
        assert result_path.is_file()


# ---------------------------------------------------------------------------
# Integration with align_datasets
# ---------------------------------------------------------------------------


class TestIntegrationWithAlignDatasets:
    """Writer output can be consumed by align_datasets."""

    _ID_REGEX = r"^(?P<scene>[^/]+)/(?P<frame>\d+)\.\w+$"
    _HIERARCHY_REGEX = r"^(?P<scene>[^/]+)/(?P<frame>\d+)\.\w+$"

    def _make_rgb_index(self, tmp_path: Path) -> dict[str, Any]:
        root = tmp_path / "rgb"
        for f in ("scene01/001.jpg", "scene01/002.jpg", "scene02/001.jpg"):
            touch(root / f)
        return index_dataset({
            "name": "rgb",
            "path": str(root),
            "type": "rgb",
            "file_extensions": [".jpg"],
            "id_regex": self._ID_REGEX,
            "hierarchy_regex": self._HIERARCHY_REGEX,
            "named_capture_group_value_separator": ":",
            "properties": {
                "euler_train": {"used_as": "input", "modality_type": "rgb"},
                "meta": {"range": [0, 255]},
            },
        })

    def test_align_with_writer_output(self, tmp_path: Path) -> None:
        rgb_index = self._make_rgb_index(tmp_path)

        writer = _make_writer(tmp_path)
        # euler-loading's full_id includes the ds-crawler file ID as the
        # leaf component.  With named id_regex groups and join_char="+",
        # the IDs look like "scene-scene01+frame-001".
        writer.get_path("/scene:scene01/frame:001/scene-scene01+frame-001", "001.png")
        writer.get_path("/scene:scene01/frame:002/scene-scene01+frame-002", "002.png")
        writer.get_path("/scene:scene02/frame:001/scene-scene02+frame-001", "001.png")

        seg_index = writer.build_output()

        aligned = align_datasets(
            {"modality": "rgb", "source": rgb_index},
            {"modality": "seg", "source": seg_index},
        )

        # All three IDs should be present in both modalities
        assert len(aligned) == 3
        for file_id, modalities in aligned.items():
            assert "rgb" in modalities
            assert "seg" in modalities
