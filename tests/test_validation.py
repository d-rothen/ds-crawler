from __future__ import annotations

from pathlib import Path

from ds_crawler import create_dataset_splits, get_dataset_contract, index_dataset_from_files
from ds_crawler.validation import validate_dataset, validate_output, validate_split_artifact

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
