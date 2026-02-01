"""Tests for ZIP archive support across handlers, config, and parser."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.config import CONFIG_FILENAME, Config, DatasetConfig, load_dataset_config
from ds_crawler.handlers import ZipHandler, get_handler
from ds_crawler.parser import DatasetParser, index_dataset, index_dataset_from_path
from ds_crawler.zip_utils import (
    _detect_root_prefix,
    get_zip_root_prefix,
    is_zip_path,
    read_json_from_zip,
    write_json_to_zip,
)

from .conftest import (
    create_depth_predictions_tree,
    create_vkitti2_tree,
    create_zip_from_tree,
    create_zip_from_tree_with_config,
    make_depth_predictions_config,
    make_vkitti2_config,
    touch,
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


def _make_simple_zip(tmp_path: Path, files: list[str]) -> Path:
    """Create a zip with empty files at the given entry paths."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.writestr(f, b"")
    return zip_path


# ===================================================================
# zip_utils
# ===================================================================


class TestIsZipPath:
    def test_existing_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hi")
        assert is_zip_path(zp) is True

    def test_nonexistent_zip(self, tmp_path: Path) -> None:
        assert is_zip_path(tmp_path / "missing.zip") is False

    def test_directory_not_zip(self, tmp_path: Path) -> None:
        assert is_zip_path(tmp_path) is False

    def test_non_zip_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "data.tar.gz"
        p.touch()
        assert is_zip_path(p) is False

    def test_string_path(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hi")
        assert is_zip_path(str(zp)) is True


class TestReadJsonFromZip:
    def test_reads_existing_entry(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        payload = {"key": "value", "num": 42}
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("config.json", json.dumps(payload))

        result = read_json_from_zip(zp, "config.json")
        assert result == payload

    def test_returns_none_for_missing_entry(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("other.txt", "hi")

        assert read_json_from_zip(zp, "config.json") is None


class TestWriteJsonToZip:
    def test_creates_new_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "new.zip"
        write_json_to_zip(zp, "output.json", {"hello": "world"})

        assert zp.exists()
        result = read_json_from_zip(zp, "output.json")
        assert result == {"hello": "world"}

    def test_adds_entry_to_existing_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("existing.txt", "keep me")

        write_json_to_zip(zp, "output.json", {"added": True})

        with zipfile.ZipFile(zp, "r") as zf:
            assert "existing.txt" in zf.namelist()
            assert "output.json" in zf.namelist()
            assert zf.read("existing.txt") == b"keep me"

        assert read_json_from_zip(zp, "output.json") == {"added": True}

    def test_replaces_existing_entry(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("output.json", json.dumps({"version": 1}))
            zf.writestr("other.txt", "keep")

        write_json_to_zip(zp, "output.json", {"version": 2})

        result = read_json_from_zip(zp, "output.json")
        assert result == {"version": 2}

        # No duplicate entries
        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.namelist().count("output.json") == 1
            assert "other.txt" in zf.namelist()


# ===================================================================
# ZipHandler
# ===================================================================


class TestZipHandler:
    def _make_config(self, path: str, **overrides) -> DatasetConfig:
        defaults = {
            "name": "test",
            "path": path,
            "type": "rgb",
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png|jpg)$",
            "id_regex": r"^(?P<f>.+)\.\w+$",
        }
        defaults.update(overrides)
        return DatasetConfig(**defaults)

    def test_finds_matching_files(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, [
            "img1.png", "img2.jpg", "data.csv",
        ])
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["img1.png", "img2.jpg"]

    def test_ignores_non_matching_extensions(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, [
            "img.png", "data.csv", "readme.txt",
        ])
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        assert files[0].name == "img.png"

    def test_nested_entries(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, [
            "a/b/c/deep.png", "shallow.png",
        ])
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["deep.png", "shallow.png"]

    def test_skips_directory_entries(self, tmp_path: Path) -> None:
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dir/", "")
            zf.writestr("dir/file.png", b"")
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        assert files[0].name == "file.png"

    def test_custom_extensions(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, [
            "file.npy", "file.png", "file.exr",
        ])
        config = self._make_config(
            str(zp), type="depth", file_extensions=[".npy", ".exr"],
        )
        handler = ZipHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["file.exr", "file.npy"]

    def test_nonexistent_zip_yields_nothing(self, tmp_path: Path) -> None:
        config = self._make_config(str(tmp_path / "nonexistent.zip"))
        handler = ZipHandler(config)
        files = list(handler.get_files())
        assert files == []

    def test_empty_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "empty.zip"
        with zipfile.ZipFile(zp, "w"):
            pass
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())
        assert files == []

    def test_case_insensitive_extension_match(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, ["image.PNG", "image.Jpg"])
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())
        assert len(files) == 2

    def test_relative_to_base_path_works(self, tmp_path: Path) -> None:
        """Synthetic paths support relative_to(base_path) correctly."""
        zp = _make_simple_zip(tmp_path, ["sub/dir/file.png"])
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        relative = files[0].relative_to(handler.base_path)
        assert str(relative) == "sub/dir/file.png"
        assert files[0].name == "file.png"


# ===================================================================
# get_handler with zip path
# ===================================================================


class TestGetHandlerZip:
    def test_returns_zip_handler_for_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "")
        assert get_handler("anything", path=str(zp)) is ZipHandler

    def test_returns_generic_for_directory(self, tmp_path: Path) -> None:
        from ds_crawler.handlers import GenericHandler

        assert get_handler("anything", path=str(tmp_path)) is GenericHandler

    def test_returns_generic_when_no_path(self) -> None:
        from ds_crawler.handlers import GenericHandler

        assert get_handler("anything") is GenericHandler


# ===================================================================
# Config loading from ZIP
# ===================================================================


class TestLoadConfigFromZip:
    def test_loads_ds_crawler_json_from_zip(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "scene" / "001.png")

        config_dict = {
            "name": "zip_cfg_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
        }

        zp = create_zip_from_tree_with_config(root, tmp_path / "ds.zip", config_dict)

        ds_config = load_dataset_config({"path": str(zp)})
        assert ds_config.name == "zip_cfg_test"

    def test_missing_config_in_zip_raises(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, ["file.png"])

        with pytest.raises(FileNotFoundError, match=CONFIG_FILENAME):
            load_dataset_config({"path": str(zp)})

    def test_inline_config_skips_zip_read(self, tmp_path: Path) -> None:
        """Full inline config does not try to read from the zip."""
        zp = _make_simple_zip(tmp_path, ["file.png"])

        ds_config = load_dataset_config({
            "name": "inline",
            "path": str(zp),
            "type": "rgb",
            "basename_regex": r"^(?P<f>.+)\.png$",
            "id_regex": r"^(?P<f>.+)\.png$",
        })
        assert ds_config.name == "inline"


# ===================================================================
# index_dataset with zip path
# ===================================================================


class TestIndexDatasetZip:
    def test_parses_zip_dataset(self, depth_predictions_zip: Path) -> None:
        config_dict = make_depth_predictions_config(str(depth_predictions_zip))
        result = index_dataset(config_dict)

        assert result["name"] == "depth_predictions"
        files = collect_all_files(result["dataset"])
        assert len(files) == 4  # 2 frames × 2 extensions

    def test_save_index_writes_into_zip(self, depth_predictions_zip: Path) -> None:
        config_dict = make_depth_predictions_config(str(depth_predictions_zip))
        result = index_dataset(config_dict, save_index=True)

        saved = read_json_from_zip(depth_predictions_zip, "output.json")
        assert saved is not None
        assert saved["name"] == result["name"]
        assert saved["dataset"] == result["dataset"]

    def test_save_index_preserves_existing_entries(
        self, depth_predictions_zip: Path
    ) -> None:
        config_dict = make_depth_predictions_config(str(depth_predictions_zip))
        index_dataset(config_dict, save_index=True)

        # Original data files should still be present
        with zipfile.ZipFile(depth_predictions_zip, "r") as zf:
            names = zf.namelist()
            assert "Scene01/00001_pred.png" in names
            assert "output.json" in names


# ===================================================================
# index_dataset_from_path with zip
# ===================================================================


class TestIndexDatasetFromPathZip:
    def test_basic_from_zip_path(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "scene" / "001.png")

        config_dict = {
            "name": "from_zip_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(root, tmp_path / "ds.zip", config_dict)

        result = index_dataset_from_path(zp)
        assert result["name"] == "from_zip_test"
        files = collect_all_files(result["dataset"])
        assert len(files) == 1

    def test_save_index_into_zip(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "001.png")

        config_dict = {
            "name": "save_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(root, tmp_path / "ds.zip", config_dict)

        result = index_dataset_from_path(zp, save_index=True)

        saved = read_json_from_zip(zp, "output.json")
        assert saved is not None
        assert saved["name"] == result["name"]

    def test_cached_output_returned(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "001.png")

        config_dict = {
            "name": "cache_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(root, tmp_path / "ds.zip", config_dict)

        # First call: index and save
        first = index_dataset_from_path(zp, save_index=True)

        # Second call: should return cached
        second = index_dataset_from_path(zp)
        assert second == first

    def test_force_reindex_ignores_cache(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "001.png")

        config_dict = {
            "name": "reindex_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(root, tmp_path / "ds.zip", config_dict)

        # Plant a stale cached output inside the zip
        write_json_to_zip(zp, "output.json", {"name": "stale", "stale": True})

        # force_reindex should ignore the stale cache
        result = index_dataset_from_path(zp, force_reindex=True)
        assert result["name"] == "reindex_test"
        assert "stale" not in result

    def test_missing_config_in_zip_raises(self, tmp_path: Path) -> None:
        zp = _make_simple_zip(tmp_path, ["file.png"])

        with pytest.raises(FileNotFoundError, match=CONFIG_FILENAME):
            index_dataset_from_path(zp)


# ===================================================================
# write_outputs_per_dataset with zip
# ===================================================================


class TestWriteOutputsPerDatasetZip:
    def test_writes_output_into_zip(self, depth_predictions_zip: Path) -> None:
        cfg_data = make_depth_predictions_config(str(depth_predictions_zip))
        # Clear output_json so it writes to the dataset path
        cfg_data["output_json"] = None
        ds_config = DatasetConfig(**cfg_data)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        paths = parser.write_outputs_per_dataset()

        assert len(paths) == 1
        assert paths[0] == depth_predictions_zip

        saved = read_json_from_zip(depth_predictions_zip, "output.json")
        assert saved is not None
        assert saved["name"] == "depth_predictions"

    def test_custom_output_json_overrides_zip(
        self, depth_predictions_zip: Path, tmp_path: Path
    ) -> None:
        """When output_json is set it writes to that filesystem path, not the zip."""
        custom_output = tmp_path / "custom_output.json"
        cfg_data = make_depth_predictions_config(str(depth_predictions_zip))
        cfg_data["output_json"] = str(custom_output)
        ds_config = DatasetConfig(**cfg_data)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        paths = parser.write_outputs_per_dataset()

        assert paths[0] == custom_output
        assert custom_output.exists()
        # The zip should NOT contain output.json
        assert read_json_from_zip(depth_predictions_zip, "output.json") is None

    def test_preserves_existing_zip_entries(
        self, depth_predictions_zip: Path
    ) -> None:
        cfg_data = make_depth_predictions_config(str(depth_predictions_zip))
        cfg_data["output_json"] = None
        ds_config = DatasetConfig(**cfg_data)
        config = Config(datasets=[ds_config])
        parser = DatasetParser(config)

        parser.write_outputs_per_dataset()

        with zipfile.ZipFile(depth_predictions_zip, "r") as zf:
            names = zf.namelist()
            assert "Scene01/00001_pred.png" in names
            assert "Scene01/00001_pred.npy" in names
            assert "output.json" in names


# ===================================================================
# End-to-end: zip produces same output as directory
# ===================================================================


class TestZipMatchesDirectory:
    """Parse from both a directory and its zip and verify identical output."""

    def _strip_ordering(self, obj: Any) -> Any:
        """Normalize file lists by sorting them by path."""
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

    def test_depth_predictions_zip_matches_dir(self, tmp_path: Path) -> None:
        # Directory version
        dir_root = tmp_path / "dir_ds"
        create_depth_predictions_tree(dir_root)
        dir_cfg = make_depth_predictions_config(str(dir_root))
        dir_ds = DatasetConfig(**dir_cfg)
        dir_parser = DatasetParser(Config(datasets=[dir_ds]))
        dir_results = dir_parser.parse_all()

        # Zip version
        zip_path = tmp_path / "ds.zip"
        create_zip_from_tree(dir_root, zip_path)
        zip_cfg = make_depth_predictions_config(str(zip_path))
        zip_ds = DatasetConfig(**zip_cfg)
        zip_parser = DatasetParser(Config(datasets=[zip_ds]))
        zip_results = zip_parser.parse_all()

        actual = self._strip_ordering(zip_results[0])
        expected = self._strip_ordering(dir_results[0])

        assert actual["name"] == expected["name"]
        assert actual["type"] == expected["type"]
        assert actual["dataset"] == expected["dataset"]

    def test_vkitti2_zip_matches_dir(self, tmp_path: Path) -> None:
        # Directory version
        dir_root = tmp_path / "dir_vk"
        create_vkitti2_tree(dir_root)
        dir_cfg = make_vkitti2_config(str(dir_root))
        dir_ds = DatasetConfig(**dir_cfg)
        dir_parser = DatasetParser(Config(datasets=[dir_ds]))
        dir_results = dir_parser.parse_all()

        # Zip version
        zip_path = tmp_path / "vk.zip"
        create_zip_from_tree(dir_root, zip_path)
        zip_cfg = make_vkitti2_config(str(zip_path))
        zip_ds = DatasetConfig(**zip_cfg)
        zip_parser = DatasetParser(Config(datasets=[zip_ds]))
        zip_results = zip_parser.parse_all()

        actual = self._strip_ordering(zip_results[0])
        expected = self._strip_ordering(dir_results[0])

        assert actual["name"] == expected["name"]
        assert actual["dataset"] == expected["dataset"]


# ===================================================================
# Root-prefix detection
# ===================================================================


class TestDetectRootPrefix:
    def test_no_entries(self) -> None:
        assert _detect_root_prefix([]) == ""

    def test_flat_entries(self) -> None:
        assert _detect_root_prefix(["a.png", "b.png"]) == ""

    def test_single_root_dir(self) -> None:
        namelist = ["mydata/", "mydata/a.png", "mydata/sub/b.png"]
        assert _detect_root_prefix(namelist) == "mydata/"

    def test_multiple_root_dirs(self) -> None:
        namelist = ["dir_a/file.png", "dir_b/file.png"]
        assert _detect_root_prefix(namelist) == ""

    def test_ignores_macosx(self) -> None:
        namelist = [
            "mydata/a.png",
            "mydata/sub/b.png",
            "__MACOSX/mydata/._a.png",
            "__MACOSX/._mydata",
        ]
        assert _detect_root_prefix(namelist) == "mydata/"

    def test_only_macosx_entries(self) -> None:
        namelist = ["__MACOSX/._something"]
        assert _detect_root_prefix(namelist) == ""

    def test_single_file_no_prefix(self) -> None:
        assert _detect_root_prefix(["file.png"]) == ""

    def test_directory_entry_only(self) -> None:
        assert _detect_root_prefix(["folder/"]) == ""

    def test_get_zip_root_prefix_function(self, tmp_path: Path) -> None:
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("root/a.png", b"")
            zf.writestr("root/sub/b.png", b"")
        assert get_zip_root_prefix(zp) == "root/"


# ===================================================================
# Root-prefix: read/write JSON
# ===================================================================


class TestReadWriteJsonWithPrefix:
    def test_read_json_with_prefix(self, tmp_path: Path) -> None:
        # Zip name matches prefix → prefix is detected
        zp = tmp_path / "mydata.zip"
        payload = {"key": "value"}
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/config.json", json.dumps(payload))
            zf.writestr("mydata/file.png", b"")
        assert read_json_from_zip(zp, "config.json") == payload

    def test_read_json_no_prefix_when_stem_mismatch(self, tmp_path: Path) -> None:
        """Prefix is not applied when zip stem differs from the prefix dir."""
        zp = tmp_path / "data.zip"
        payload = {"key": "value"}
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/config.json", json.dumps(payload))
            zf.writestr("mydata/file.png", b"")
        # Stem is "data", prefix is "mydata/" → no match
        assert read_json_from_zip(zp, "config.json") is None

    def test_read_json_prefers_exact_match(self, tmp_path: Path) -> None:
        """If both bare and prefixed entries exist, exact match wins."""
        zp = tmp_path / "mydata.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("config.json", json.dumps({"exact": True}))
            zf.writestr("mydata/config.json", json.dumps({"prefixed": True}))
            zf.writestr("mydata/file.png", b"")
        assert read_json_from_zip(zp, "config.json") == {"exact": True}

    def test_write_json_into_prefixed_zip(self, tmp_path: Path) -> None:
        zp = tmp_path / "mydata.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/file.png", b"")
        write_json_to_zip(zp, "output.json", {"written": True})
        # Should be stored with prefix
        with zipfile.ZipFile(zp, "r") as zf:
            assert "mydata/output.json" in zf.namelist()
        assert read_json_from_zip(zp, "output.json") == {"written": True}

    def test_write_no_prefix_when_stem_mismatch(self, tmp_path: Path) -> None:
        """Write without prefix when zip stem differs from detected prefix."""
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/file.png", b"")
        write_json_to_zip(zp, "output.json", {"written": True})
        # Stem is "data", prefix is "mydata/" → no match, written bare
        with zipfile.ZipFile(zp, "r") as zf:
            assert "output.json" in zf.namelist()
            assert "mydata/output.json" not in zf.namelist()

    def test_write_replaces_prefixed_entry(self, tmp_path: Path) -> None:
        zp = tmp_path / "mydata.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/output.json", json.dumps({"v": 1}))
            zf.writestr("mydata/file.png", b"")
        write_json_to_zip(zp, "output.json", {"v": 2})
        assert read_json_from_zip(zp, "output.json") == {"v": 2}
        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.namelist().count("mydata/output.json") == 1


# ===================================================================
# Root-prefix: ZipHandler
# ===================================================================


class TestZipHandlerWithPrefix:
    def _make_config(self, path: str, **overrides) -> DatasetConfig:
        defaults = {
            "name": "test",
            "path": path,
            "type": "rgb",
            "basename_regex": r"^(?P<f>.+)\.(?P<ext>png|jpg)$",
            "id_regex": r"^(?P<f>.+)\.\w+$",
        }
        defaults.update(overrides)
        return DatasetConfig(**defaults)

    def test_strips_root_prefix(self, tmp_path: Path) -> None:
        # Zip name matches prefix dir → stripping happens
        zp = tmp_path / "mydata.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/sub/img1.png", b"")
            zf.writestr("mydata/img2.png", b"")
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        relatives = sorted(str(f.relative_to(handler.base_path)) for f in files)
        assert relatives == ["img2.png", "sub/img1.png"]

    def test_no_strip_when_stem_mismatch(self, tmp_path: Path) -> None:
        """Prefix is NOT stripped when zip stem differs from prefix dir."""
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/sub/img1.png", b"")
            zf.writestr("mydata/img2.png", b"")
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        relatives = sorted(str(f.relative_to(handler.base_path)) for f in files)
        assert relatives == ["mydata/img2.png", "mydata/sub/img1.png"]

    def test_skips_macosx_entries(self, tmp_path: Path) -> None:
        zp = tmp_path / "mydata.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("mydata/img.png", b"")
            zf.writestr("__MACOSX/mydata/._img.png", b"")
            zf.writestr("__MACOSX/._mydata", b"")
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        assert files[0].name == "img.png"

    def test_no_prefix_still_works(self, tmp_path: Path) -> None:
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("sub/img.png", b"")
            zf.writestr("other/img2.png", b"")
        config = self._make_config(str(zp))
        handler = ZipHandler(config)
        files = list(handler.get_files())

        relatives = sorted(str(f.relative_to(handler.base_path)) for f in files)
        assert relatives == ["other/img2.png", "sub/img.png"]


# ===================================================================
# Root-prefix: end-to-end parsing
# ===================================================================


class TestPrefixedZipMatchesDirectory:
    """Prefixed zips should produce the same output as flat zips / dirs."""

    def _strip_ordering(self, obj: Any) -> Any:
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

    def test_depth_predictions_prefixed_zip_matches_dir(
        self, tmp_path: Path
    ) -> None:
        # Directory version
        dir_root = tmp_path / "dir_ds"
        create_depth_predictions_tree(dir_root)
        dir_cfg = make_depth_predictions_config(str(dir_root))
        dir_ds = DatasetConfig(**dir_cfg)
        dir_parser = DatasetParser(Config(datasets=[dir_ds]))
        dir_results = dir_parser.parse_all()

        # Prefixed zip: name "dp" matches prefix "dp/"
        zip_path = tmp_path / "dp.zip"
        create_zip_from_tree(dir_root, zip_path, root_prefix="dp/")
        zip_cfg = make_depth_predictions_config(str(zip_path))
        zip_ds = DatasetConfig(**zip_cfg)
        zip_parser = DatasetParser(Config(datasets=[zip_ds]))
        zip_results = zip_parser.parse_all()

        actual = self._strip_ordering(zip_results[0])
        expected = self._strip_ordering(dir_results[0])

        assert actual["name"] == expected["name"]
        assert actual["dataset"] == expected["dataset"]

    def test_vkitti2_prefixed_zip_matches_dir(self, tmp_path: Path) -> None:
        # Directory version
        dir_root = tmp_path / "dir_vk"
        create_vkitti2_tree(dir_root)
        dir_cfg = make_vkitti2_config(str(dir_root))
        dir_ds = DatasetConfig(**dir_cfg)
        dir_parser = DatasetParser(Config(datasets=[dir_ds]))
        dir_results = dir_parser.parse_all()

        # Prefixed zip: name "vkitti2" matches prefix "vkitti2/"
        zip_path = tmp_path / "vkitti2.zip"
        create_zip_from_tree(dir_root, zip_path, root_prefix="vkitti2/")
        zip_cfg = make_vkitti2_config(str(zip_path))
        zip_ds = DatasetConfig(**zip_cfg)
        zip_parser = DatasetParser(Config(datasets=[zip_ds]))
        zip_results = zip_parser.parse_all()

        actual = self._strip_ordering(zip_results[0])
        expected = self._strip_ordering(dir_results[0])

        assert actual["name"] == expected["name"]
        assert actual["dataset"] == expected["dataset"]

    def test_index_from_path_with_prefixed_config(self, tmp_path: Path) -> None:
        """index_dataset_from_path loads ds-crawler.json from a prefixed zip."""
        root = tmp_path / "_tree"
        touch(root / "scene" / "001.png")

        config_dict = {
            "name": "prefixed_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<dir>[^/]+)/(?P<frame>\d+)\.png$",
        }
        # Zip name "mydata" matches prefix "mydata/"
        zp = create_zip_from_tree_with_config(
            root, tmp_path / "mydata.zip", config_dict, root_prefix="mydata/"
        )

        result = index_dataset_from_path(zp)
        assert result["name"] == "prefixed_test"
        files = collect_all_files(result["dataset"])
        assert len(files) == 1

    def test_save_index_into_prefixed_zip(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "001.png")

        config_dict = {
            "name": "save_prefix_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(
            root, tmp_path / "mydata.zip", config_dict, root_prefix="mydata/"
        )

        result = index_dataset_from_path(zp, save_index=True)

        # output.json should be readable (via prefix detection)
        saved = read_json_from_zip(zp, "output.json")
        assert saved is not None
        assert saved["name"] == result["name"]

        # Verify it was stored under the prefix
        with zipfile.ZipFile(zp, "r") as zf:
            assert "mydata/output.json" in zf.namelist()

    def test_cached_output_in_prefixed_zip(self, tmp_path: Path) -> None:
        root = tmp_path / "_tree"
        touch(root / "001.png")

        config_dict = {
            "name": "cache_prefix_test",
            "path": "PLACEHOLDER",
            "type": "rgb",
            "file_extensions": [".png"],
            "basename_regex": r"^(?P<frame>\d+)\.(?P<ext>png)$",
            "id_regex": r"^(?P<frame>\d+)\.png$",
        }
        zp = create_zip_from_tree_with_config(
            root, tmp_path / "mydata.zip", config_dict, root_prefix="mydata/"
        )

        first = index_dataset_from_path(zp, save_index=True)
        second = index_dataset_from_path(zp)
        assert second == first
