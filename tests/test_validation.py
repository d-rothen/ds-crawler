from __future__ import annotations

from pathlib import Path

from ds_crawler import create_dataset_splits, get_dataset_contract, index_dataset_from_files
from ds_crawler.validation import validate_dataset, validate_output, validate_split_artifact
import pytest

from .current_helpers import create_files, sample_config, write_crawler_metadata


def test_validate_output_accepts_current_output_shape(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene/0001.png"])
    output = index_dataset_from_files(
        sample_config(root, extensions=[".png"], id_regex=r"^(.+)\.png$"),
        files,
        base_path=root,
    )

    validated = validate_output(output)

    assert validated["contract"]["kind"] == "dataset_index"


def test_validate_dataset_reads_canonical_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    create_files(root, ["scene/0001.png", "scene/0002.png"])
    config = sample_config(root, extensions=[".png"], id_regex=r"^(.+)\.png$")
    write_crawler_metadata(root, config)

    output = index_dataset_from_files(config, list(root.rglob("*.png")), base_path=root)
    create_files(root, [])  # keep root creation explicit for readability
    from ds_crawler.artifacts import save_output_artifacts

    save_output_artifacts(root, output)
    result = validate_dataset(root)

    assert result["has_head"] is True
    assert result["has_index"] is True
    assert result["output"]["head"]["dataset"]["id"] == "demo_rgb"


def test_get_dataset_contract_and_validate_split_artifact(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    create_files(root, ["scene/0001.png", "scene/0002.png"])
    config = sample_config(root, extensions=[".png"], id_regex=r"^(.+)\.png$")
    write_crawler_metadata(root, config)

    from ds_crawler.parser import index_dataset_from_path

    index_dataset_from_path(root, save_index=True)
    create_dataset_splits(root, ["train"], [1.0], seed=3)

    contract = get_dataset_contract(root)
    assert contract.dataset_id == "demo_rgb"
    assert contract.get_namespace("euler_train")["slot"] == "demo.input.rgb"

    split_path = root / ".ds_crawler" / "split_train.json"
    import json

    with open(split_path) as f:
        split_artifact = json.load(f)
    assert validate_split_artifact(split_artifact)["split"]["name"] == "train"


def test_validate_split_artifact_rejects_missing_source_index_file() -> None:
    with pytest.raises(ValueError, match="split.source_index_file"):
        validate_split_artifact(
            {
                "contract": {"kind": "dataset_split", "version": "1.0"},
                "split": {"name": "train"},
                "index": {"files": []},
            }
        )


def _output_with_attributes(attributes: object) -> dict:
    return {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "head_file": "dataset-head.json",
        "head": {
            "contract": {"kind": "dataset_head", "version": "1.0"},
            "dataset": {"id": "x", "name": "X"},
            "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
            "addons": {},
        },
        "index": {
            "files": [
                {
                    "id": "0001",
                    "path": "0001.png",
                    "path_properties": {},
                    "basename_properties": {},
                    "attributes": attributes,
                }
            ]
        },
    }


def test_validate_output_accepts_attributes_dict() -> None:
    out = _output_with_attributes({"k": "v", "n": 7, "nested": {"a": 1}})
    validated = validate_output(out)
    assert validated["index"]["files"][0]["attributes"]["nested"] == {"a": 1}


def test_validate_output_rejects_non_dict_attributes() -> None:
    with pytest.raises(ValueError, match="attributes must be an object"):
        validate_output(_output_with_attributes(["not", "a", "dict"]))


def test_validate_output_rejects_non_string_attribute_keys() -> None:
    out = _output_with_attributes({1: "v"})
    with pytest.raises(ValueError, match="attributes keys must be strings"):
        validate_output(out)
