"""Tests for crawler.handlers – file discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from crawler.config import DatasetConfig
from crawler.handlers import GenericHandler, get_handler, HANDLERS
from .conftest import touch


# ===================================================================
# get_handler registry
# ===================================================================


class TestGetHandler:
    def test_unknown_name_returns_generic(self) -> None:
        assert get_handler("unknown_dataset") is GenericHandler

    def test_generic_handler_is_default(self) -> None:
        assert get_handler("anything") is GenericHandler


# ===================================================================
# GenericHandler
# ===================================================================


class TestGenericHandler:
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
        root = tmp_path / "data"
        touch(root / "img1.png")
        touch(root / "img2.jpg")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["img1.png", "img2.jpg"]

    def test_ignores_non_matching_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "img.png")
        touch(root / "data.csv")
        touch(root / "readme.txt")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        assert files[0].name == "img.png"

    def test_recursive_discovery(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a" / "b" / "c" / "deep.png")
        touch(root / "shallow.png")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["deep.png", "shallow.png"]

    def test_custom_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.npy")
        touch(root / "file.png")
        touch(root / "file.exr")

        config = self._make_config(
            str(root), type="depth", file_extensions=[".npy", ".exr"]
        )
        handler = GenericHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["file.exr", "file.npy"]

    def test_nonexistent_path_yields_nothing(self, tmp_path: Path) -> None:
        config = self._make_config(str(tmp_path / "nonexistent"))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        assert files == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        assert files == []

    def test_case_insensitive_extension_match(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "image.PNG")
        touch(root / "image.Jpg")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        # GenericHandler compares suffix.lower() against extensions
        assert len(files) == 2

    def test_default_rgb_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a.png")
        touch(root / "b.jpg")
        touch(root / "c.jpeg")
        touch(root / "d.tiff")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["a.png", "b.jpg", "c.jpeg"]

    def test_default_depth_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "a.png")
        touch(root / "b.exr")
        touch(root / "c.npy")
        touch(root / "d.pfm")
        touch(root / "e.jpg")

        config = self._make_config(str(root), type="depth")
        handler = GenericHandler(config)
        files = list(handler.get_files())

        names = sorted(f.name for f in files)
        assert names == ["a.png", "b.exr", "c.npy", "d.pfm"]

    def test_returns_absolute_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "data"
        touch(root / "file.png")

        config = self._make_config(str(root))
        handler = GenericHandler(config)
        files = list(handler.get_files())

        assert len(files) == 1
        assert files[0].is_absolute()

    def test_base_path_set_from_config(self, tmp_path: Path) -> None:
        config = self._make_config(str(tmp_path / "mydata"))
        handler = GenericHandler(config)
        assert handler.base_path == tmp_path / "mydata"
