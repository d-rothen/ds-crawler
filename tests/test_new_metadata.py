from __future__ import annotations

import json

from ds_crawler import DatasetWriter, get_dataset_contract, index_dataset_from_files
from ds_crawler.migration import migrate_dataset_metadata


def _sample_head() -> dict:
    return {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "demo_rgb", "name": "Demo RGB"},
        "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
        "addons": {
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": "demo.input.rgb",
            }
        },
    }


def test_index_dataset_from_files_emits_head_and_index(tmp_path) -> None:
    root = tmp_path / "rgb"
    root.mkdir()
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head": _sample_head(),
        "source": {"path": str(root)},
        "indexing": {
            "id": {"regex": "^(.+)$", "join_char": "+"},
            "constraints": {"flat_ids_unique": True},
        },
    }

    output = index_dataset_from_files(
        config,
        [root / "0001.png", root / "0002.png"],
        base_path=root,
    )

    assert output["contract"]["kind"] == "dataset_index"
    assert output["head"]["dataset"]["id"] == "demo_rgb"
    assert output["index"]["files"][0]["path"] == "0001.png"
    assert output["head"]["modality"]["meta"]["file_types"] == ["png"]


def test_dataset_writer_writes_separate_dataset_head(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "predictions",
        head=_sample_head(),
    )
    path = writer.get_path("/scene:01/0001", "0001.png")
    path.write_bytes(b"data")
    writer.save_index()

    with open(tmp_path / "predictions" / ".ds_crawler" / "dataset-head.json") as f:
        head = json.load(f)
    with open(tmp_path / "predictions" / ".ds_crawler" / "output.json") as f:
        output = json.load(f)

    assert head["dataset"]["id"] == "demo_rgb"
    assert output["head"]["dataset"]["id"] == "demo_rgb"
    assert output["index"]["children"]["scene:01"]["files"][0]["path"] == "01/0001.png"


def test_migrate_metadata_rewrites_legacy_files(tmp_path) -> None:
    legacy_config = {
        "name": "Legacy RGB",
        "path": str(tmp_path),
        "type": "rgb",
        "id_regex": "^(.+)$",
        "properties": {
            "meta": {"range": [0, 255]},
            "euler_train": {
                "used_as": "input",
                "slot": "demo.input.rgb",
                "modality_type": "rgb",
            },
            "gt": False,
        },
    }
    legacy_output = {
        "dataset_contract_version": "1.0",
        "name": "Legacy RGB",
        "type": "rgb",
        "id_regex": "^(.+)$",
        "meta": {"range": [0, 255], "file_types": ["png"]},
        "euler_train": {
            "used_as": "input",
            "slot": "demo.input.rgb",
            "modality_type": "rgb",
        },
        "gt": False,
        "dataset": {"files": [{"path": "0001.png", "id": "0001.png"}]},
    }
    (tmp_path / "0001.png").write_bytes(b"data")
    with open(tmp_path / "ds-crawler.json", "w") as f:
        json.dump(legacy_config, f)
    with open(tmp_path / "output.json", "w") as f:
        json.dump(legacy_output, f)

    result = migrate_dataset_metadata(tmp_path)
    contract = get_dataset_contract(tmp_path)

    assert result["wrote_head"] is True
    assert contract.dataset_id == "legacy_rgb"
    assert contract.type == "rgb"
