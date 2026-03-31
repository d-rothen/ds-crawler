from __future__ import annotations

from pathlib import Path

from ds_crawler.config import DatasetConfig
from ds_crawler.handlers.generic import GenericHandler
from ds_crawler.handlers.zip_handler import ZipHandler

from .current_helpers import create_files, sample_config, zip_tree


def test_generic_handler_filters_extensions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    create_files(root, ["scene/0001.png", "scene/0002.txt"])

    config = DatasetConfig.from_dict(
        sample_config(
            root,
            extensions=[".png"],
            id_regex=r"^(.+)\.png$",
        )
    )

    files = sorted(path.relative_to(root) for path in GenericHandler(config).get_files())

    assert files == [Path("scene/0001.png")]


def test_zip_handler_strips_matching_root_prefix(tmp_path: Path) -> None:
    root = tmp_path / "dataset_tree"
    create_files(root, ["scene/0001.png", "scene/0002.png"])
    zip_path = zip_tree(root, tmp_path / "dataset_tree.zip", prefix="dataset_tree/")

    config = DatasetConfig.from_dict(
        sample_config(
            zip_path,
            extensions=[".png"],
            id_regex=r"^(.+)\.png$",
        )
    )

    files = sorted(path.relative_to(zip_path) for path in ZipHandler(config).get_files())

    assert files == [Path("scene/0001.png"), Path("scene/0002.png")]
