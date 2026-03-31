from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from ds_crawler._dataset_contract import build_default_meta


def _slugify(value: str) -> str:
    return "_".join(token for token in value.lower().replace("-", " ").split() if token)


def sample_head(
    *,
    name: str = "Demo RGB",
    modality: str = "rgb",
    dataset_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = build_default_meta(modality)
    if meta is None:
        meta = {"range": [0, 255]}

    head: dict[str, Any] = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {
            "id": dataset_id or _slugify(name),
            "name": name,
        },
        "modality": {
            "key": modality,
            "meta": meta,
        },
        "addons": {
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": f"demo.input.{modality}",
            }
        },
    }
    if attributes:
        head["dataset"]["attributes"] = dict(attributes)
    return head


def sample_config(
    path: str | Path,
    *,
    name: str = "Demo RGB",
    modality: str = "rgb",
    dataset_id: str | None = None,
    extensions: list[str] | None = None,
    id_regex: str = r"^(.+)\.png$",
    hierarchy_regex: str | None = None,
    basename_regex: str | None = None,
    path_regex: str | None = None,
    separator: str = ":",
) -> dict[str, Any]:
    indexing: dict[str, Any] = {
        "id": {
            "regex": id_regex,
            "join_char": "+",
        },
        "constraints": {
            "flat_ids_unique": True,
        },
    }
    if extensions is not None:
        indexing["files"] = {"extensions": list(extensions)}
    if hierarchy_regex is not None:
        indexing["hierarchy"] = {
            "regex": hierarchy_regex,
            "separator": separator,
        }
    if basename_regex is not None or path_regex is not None:
        indexing["properties"] = {}
        if basename_regex is not None:
            indexing["properties"]["basename"] = {"regex": basename_regex}
        if path_regex is not None:
            indexing["properties"]["path"] = {"regex": path_regex}

    return {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "name": name,
        "path": str(path),
        "head": sample_head(
            name=name,
            modality=modality,
            dataset_id=dataset_id,
        ),
        "source": {"path": str(path)},
        "indexing": indexing,
    }


def write_crawler_metadata(root: Path, config: dict[str, Any]) -> None:
    metadata_dir = root / ".ds_crawler"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    head = config["head"]
    on_disk_config = {
        key: value
        for key, value in config.items()
        if key != "head"
    }
    on_disk_config["head_file"] = "dataset-head.json"
    on_disk_config["source"] = {"path": "."}

    with open(metadata_dir / "dataset-head.json", "w") as f:
        json.dump(head, f, indent=2)
    with open(metadata_dir / "ds-crawler.json", "w") as f:
        json.dump(on_disk_config, f, indent=2)


def create_files(root: Path, relative_paths: Iterable[str], *, content: bytes = b"data") -> list[Path]:
    paths: list[Path] = []
    for relative_path in relative_paths:
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        paths.append(file_path)
    return paths


def zip_tree(root: Path, zip_path: Path, *, prefix: str = "") -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, prefix + str(path.relative_to(root)))
    return zip_path
