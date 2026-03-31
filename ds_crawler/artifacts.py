"""Helpers for persisting and hydrating dataset metadata artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import (
    CONFIG_FILENAME,
    CRAWLER_CONFIG_KIND,
    CRAWLER_CONFIG_VERSION,
    DatasetConfig,
    load_dataset_config,
)
from .zip_utils import (
    DATASET_HEAD_FILENAME,
    METADATA_DIR,
    OUTPUT_FILENAME,
    is_zip_path,
    read_metadata_json,
    write_metadata_json_batch,
)

DATASET_SPLIT_KIND = "dataset_split"
DATASET_SPLIT_VERSION = "1.0"


def build_index_artifact(output: dict[str, Any]) -> dict[str, Any]:
    """Return the minimal on-disk index artifact for a full dataset output."""
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    if "contract" not in output or "index" not in output:
        raise ValueError("output must contain contract and index")

    artifact = {
        "contract": output["contract"],
        "index": output["index"],
    }
    generator = output.get("generator")
    if generator is not None:
        artifact["generator"] = generator
    execution = output.get("execution")
    if execution is not None:
        artifact["execution"] = execution
    return artifact


def build_split_artifact(
    output: dict[str, Any],
    *,
    split_name: str,
    source_index_file: str = OUTPUT_FILENAME,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the on-disk artifact for an inline split."""
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    if "index" not in output or not isinstance(output["index"], dict):
        raise ValueError("output.index must be an object")
    if not isinstance(split_name, str) or not split_name:
        raise ValueError("split_name must be a non-empty string")
    if not isinstance(source_index_file, str) or not source_index_file:
        raise ValueError("source_index_file must be a non-empty string")

    artifact = {
        "contract": {
            "kind": DATASET_SPLIT_KIND,
            "version": DATASET_SPLIT_VERSION,
        },
        "split": {
            "name": split_name,
            "source_index_file": source_index_file,
        },
        "index": deepcopy(output["index"]),
    }
    generator = output.get("generator")
    if generator is not None:
        artifact["generator"] = deepcopy(generator)
    if execution:
        artifact["execution"] = deepcopy(execution)
    return artifact


def build_crawler_config_for_output(output: dict[str, Any]) -> dict[str, Any]:
    """Derive a crawler config mapping from a full dataset output."""
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    if "head" not in output or not isinstance(output["head"], dict):
        raise ValueError("output.head must be an object")

    head_file = output.get("head_file", DATASET_HEAD_FILENAME)
    if not isinstance(head_file, str) or not head_file:
        raise ValueError("output.head_file must be a non-empty string")

    indexing = output.get("indexing", {})
    if not isinstance(indexing, dict):
        raise ValueError("output.indexing must be an object")

    config = {
        "contract": {
            "kind": CRAWLER_CONFIG_KIND,
            "version": CRAWLER_CONFIG_VERSION,
        },
        "head_file": head_file,
        "source": {
            "path": ".",
        },
        "indexing": dict(indexing),
    }

    id_cfg = indexing.get("id")
    has_id_regex = (
        isinstance(id_cfg, dict)
        and isinstance(id_cfg.get("regex"), str)
        and bool(id_cfg.get("regex"))
    )
    if not has_id_regex:
        config["source"]["prebuilt_index_file"] = OUTPUT_FILENAME

    return config


def hydrate_index_artifact(
    index_artifact: dict[str, Any],
    ds_config: DatasetConfig,
) -> dict[str, Any]:
    """Expand an on-disk index artifact into the richer in-memory output shape."""
    if not isinstance(index_artifact, dict):
        raise ValueError("index_artifact must be an object")
    if "contract" not in index_artifact or "index" not in index_artifact:
        raise ValueError("index_artifact must contain contract and index")

    hydrated = dict(index_artifact)
    hydrated["head_file"] = ds_config.head_file
    hydrated["head"] = ds_config.head_mapping()
    hydrated["indexing"] = ds_config.to_indexing_dict()
    hydrated.setdefault("generator", {})
    hydrated.setdefault("execution", {})
    return hydrated


def hydrate_split_artifact(
    split_artifact: dict[str, Any],
    base_output: dict[str, Any],
) -> dict[str, Any]:
    """Expand a split artifact into the richer in-memory output shape."""
    if not isinstance(split_artifact, dict):
        raise ValueError("split_artifact must be an object")
    if "index" not in split_artifact or "split" not in split_artifact:
        raise ValueError("split_artifact must contain split and index")
    if not isinstance(base_output, dict):
        raise ValueError("base_output must be an object")

    hydrated = dict(base_output)
    hydrated["index"] = deepcopy(split_artifact["index"])
    hydrated["split"] = deepcopy(split_artifact["split"])

    split_execution = split_artifact.get("execution")
    if isinstance(split_execution, dict) and split_execution:
        execution = dict(hydrated.get("execution") or {})
        execution["split"] = deepcopy(split_execution)
        hydrated["execution"] = execution

    generator = split_artifact.get("generator")
    if "generator" not in hydrated and isinstance(generator, dict):
        hydrated["generator"] = deepcopy(generator)
    return hydrated


def load_saved_output(
    dataset_path: str | Path,
    *,
    filename: str = OUTPUT_FILENAME,
) -> dict[str, Any] | None:
    """Load and hydrate a persisted dataset index artifact from disk."""
    dataset_root = Path(dataset_path)
    index_artifact = read_metadata_json(dataset_root, filename)
    if index_artifact is None:
        return None
    ds_config = load_dataset_config({"path": str(dataset_root)})
    return hydrate_index_artifact(index_artifact, ds_config)


def load_prebuilt_output(ds_config: DatasetConfig) -> dict[str, Any]:
    """Load and hydrate a prebuilt index configured by ``prebuilt_index_file``."""
    if ds_config.prebuilt_index_file is None:
        raise FileNotFoundError("Dataset config has no prebuilt index file")

    prebuilt_path = Path(ds_config.prebuilt_index_file)
    if prebuilt_path.is_file():
        with open(prebuilt_path) as f:
            index_artifact = json.load(f)
    else:
        index_artifact = read_metadata_json(
            Path(ds_config.dataset_root),
            prebuilt_path.name,
        )
        if index_artifact is None:
            raise FileNotFoundError(
                f"No prebuilt index found at {ds_config.prebuilt_index_file}"
            )

    return hydrate_index_artifact(index_artifact, ds_config)


def save_output_artifacts(
    dataset_path: str | Path,
    output: dict[str, Any],
    *,
    filename: str = OUTPUT_FILENAME,
) -> Path:
    """Persist head/config/index artifacts for a dataset output."""
    dataset_root = Path(dataset_path)
    if "head" not in output or not isinstance(output["head"], dict):
        raise ValueError("output.head must be an object")

    head_file = output.get("head_file", DATASET_HEAD_FILENAME)
    if not isinstance(head_file, str) or not head_file:
        raise ValueError("output.head_file must be a non-empty string")

    write_metadata_json_batch(
        dataset_root,
        {
            head_file: output["head"],
            CONFIG_FILENAME: build_crawler_config_for_output(output),
            filename: build_index_artifact(output),
        },
    )

    if is_zip_path(dataset_root):
        return dataset_root
    return dataset_root / METADATA_DIR / filename
