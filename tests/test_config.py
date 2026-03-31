from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_crawler.config import Config, DatasetConfig, load_dataset_config

from .current_helpers import sample_config, write_crawler_metadata


def test_dataset_config_from_dict_with_embedded_head(tmp_path: Path) -> None:
    config = sample_config(
        tmp_path / "rgb",
        name="RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
    )

    dataset_config = DatasetConfig.from_dict(config)

    assert dataset_config.name == "RGB"
    assert dataset_config.type == "rgb"
    assert dataset_config.file_extensions == [".png"]
    assert dataset_config.id_regex == r"^(.+)\.png$"


def test_dataset_config_allows_prebuilt_index_without_id_regex(tmp_path: Path) -> None:
    config = sample_config(tmp_path / "rgb")
    config["indexing"].pop("id")
    config["source"]["prebuilt_index_file"] = "index.json"

    dataset_config = DatasetConfig.from_dict(config)

    assert dataset_config.id_regex is None
    assert Path(dataset_config.prebuilt_index_file).name == "index.json"


def test_dataset_config_requires_id_regex_or_prebuilt_index(tmp_path: Path) -> None:
    config = sample_config(tmp_path / "rgb")
    config["indexing"].pop("id")

    with pytest.raises(ValueError, match="indexing.id.regex is required"):
        DatasetConfig.from_dict(config)


def test_dataset_config_rejects_id_regex_without_capture_groups(tmp_path: Path) -> None:
    config = sample_config(tmp_path / "rgb", id_regex=r"^.+\.png$")

    with pytest.raises(ValueError, match="must contain at least one capture group"):
        DatasetConfig.from_dict(config)


def test_dataset_config_requires_separator_for_named_hierarchy_groups(tmp_path: Path) -> None:
    config = sample_config(
        tmp_path / "rgb",
        hierarchy_regex=r"^(?P<scene>[^/]+)/(?P<frame>\d+)\.png$",
    )
    config["indexing"]["hierarchy"].pop("separator")

    with pytest.raises(ValueError, match="indexing.hierarchy.separator"):
        DatasetConfig.from_dict(config)


def test_dataset_config_normalizes_path_filters(tmp_path: Path) -> None:
    config = sample_config(tmp_path / "rgb")
    config["indexing"]["files"] = {
        "extensions": [".png"],
        "path_filters": {
            "include_terms": ["scene"],
            "term_match_mode": "path_segment",
        },
    }

    dataset_config = DatasetConfig.from_dict(config)

    assert dataset_config.path_filters == {
        "include_terms": ["scene"],
        "term_match_mode": "path_segment",
    }


def test_load_dataset_config_from_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    config = sample_config(root, name="RGB")
    write_crawler_metadata(root, config)

    dataset_config = load_dataset_config({"path": str(root)})

    assert dataset_config.name == "RGB"
    assert dataset_config.dataset_id == "rgb"
    assert dataset_config.head_file == "dataset-head.json"


def test_dataset_config_reads_head_from_dataset_head_file(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    config = sample_config(root, name="RGB")
    write_crawler_metadata(root, config)

    on_disk_config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": config["indexing"],
    }

    dataset_config = DatasetConfig.from_dict(on_disk_config, dataset_root=root)

    assert dataset_config.name == "RGB"
    assert dataset_config.type == "rgb"


def test_config_from_file_loads_multiple_current_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    data = {
        "datasets": [
            sample_config(tmp_path / "rgb", name="RGB"),
            sample_config(
                tmp_path / "depth",
                name="Depth",
                modality="depth",
                extensions=[".npy"],
                id_regex=r"^(.+)\.npy$",
            ),
        ]
    }
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)

    config = Config.from_file(config_path)

    assert [dataset.name for dataset in config.datasets] == ["RGB", "Depth"]
    assert [dataset.type for dataset in config.datasets] == ["rgb", "depth"]
