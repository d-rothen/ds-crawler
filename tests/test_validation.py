"""Tests for public validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ds_crawler.config import CONFIG_FILENAME, DatasetConfig
from ds_crawler.parser import index_dataset
from ds_crawler.validation import (
    validate_crawler_config,
    validate_dataset,
    validate_output,
)
from ds_crawler.zip_utils import METADATA_DIR, OUTPUT_FILENAME

from .conftest import create_depth_predictions_tree, make_depth_predictions_config


def _first_file_entry(node: dict[str, Any]) -> dict[str, Any]:
    files = node.get("files")
    if files:
        return files[0]

    for child in node.get("children", {}).values():
        found = _first_file_entry(child)
        if found:
            return found

    raise AssertionError("No file entry found in output dataset node")


class TestValidateCrawlerConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))

        validated = validate_crawler_config(config)

        assert isinstance(validated, DatasetConfig)
        assert validated.name == "depth_predictions"
        assert validated.type == "depth"

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        config.pop("id_regex")

        with pytest.raises(ValueError, match="missing required field"):
            validate_crawler_config(config)


class TestValidateOutput:
    def test_single_output_object(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        output = index_dataset(config)

        validated = validate_output(output)
        assert validated is output

    def test_output_list(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        output = index_dataset(config)
        payload = [output]

        validated = validate_output(payload)
        assert validated is payload

    def test_invalid_file_entry_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        output = index_dataset(config)
        first_entry = _first_file_entry(output["dataset"])
        first_entry["id"] = 123

        with pytest.raises(ValueError, match=r"\.id must be a non-empty string"):
            validate_output(output)

    def test_euler_train_slot_is_optional(self, tmp_path: Path) -> None:
        root = tmp_path / "depth_predictions"
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        output = index_dataset(config)
        output["euler_train"].pop("slot", None)

        validated = validate_output(output)
        assert validated is output


class TestValidateDataset:
    def _build_dataset_files(self, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        create_depth_predictions_tree(root)
        config = make_depth_predictions_config(str(root))
        output = index_dataset(config)
        return config, output

    def test_validates_root_metadata_files(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset_root"
        root.mkdir(parents=True, exist_ok=True)
        config, output = self._build_dataset_files(root)

        with open(root / CONFIG_FILENAME, "w") as f:
            json.dump(config, f, indent=2)
        with open(root / OUTPUT_FILENAME, "w") as f:
            json.dump(output, f, indent=2)

        result = validate_dataset(root)

        assert result["has_config"] is True
        assert result["has_output"] is True
        assert isinstance(result["config"], DatasetConfig)
        assert result["output"]["name"] == output["name"]

    def test_validates_hidden_metadata_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset_root"
        root.mkdir(parents=True, exist_ok=True)
        config, output = self._build_dataset_files(root)

        metadata_dir = root / METADATA_DIR
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_dir / CONFIG_FILENAME, "w") as f:
            json.dump(config, f, indent=2)
        with open(metadata_dir / OUTPUT_FILENAME, "w") as f:
            json.dump(output, f, indent=2)

        result = validate_dataset(root)

        assert result["has_config"] is True
        assert result["has_output"] is True
        assert isinstance(result["config"], DatasetConfig)
        assert result["output"]["name"] == output["name"]

    def test_raises_when_no_metadata_exists(self, tmp_path: Path) -> None:
        root = tmp_path / "empty_dataset"
        root.mkdir(parents=True, exist_ok=True)

        with pytest.raises(
            FileNotFoundError, match=r"No ds-crawler\.json or output\.json found"
        ):
            validate_dataset(root)
