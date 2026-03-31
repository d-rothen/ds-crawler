from __future__ import annotations

from pathlib import Path

from ds_crawler import index_dataset_from_files, index_dataset_from_path

from .current_helpers import create_files, sample_config, write_crawler_metadata


def test_index_dataset_from_files_emits_current_output_shape(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    files = create_files(root, ["scene/0001.png", "scene/0002.png"])

    output = index_dataset_from_files(
        sample_config(
            root,
            name="RGB",
            modality="rgb",
            extensions=[".png"],
            id_regex=r"^(.+)\.png$",
        ),
        files,
        base_path=root,
    )

    assert output["contract"]["kind"] == "dataset_index"
    assert output["head"]["dataset"]["name"] == "RGB"
    assert output["head"]["modality"]["key"] == "rgb"
    assert output["head"]["modality"]["meta"]["file_types"] == ["png"]
    assert [entry["path"] for entry in output["index"]["files"]] == [
        "scene/0001.png",
        "scene/0002.png",
    ]


def test_index_dataset_from_path_writes_canonical_metadata_files(tmp_path: Path) -> None:
    root = tmp_path / "rgb"
    create_files(root, ["scene/0001.png", "scene/0002.png"])
    config = sample_config(
        root,
        name="RGB",
        modality="rgb",
        extensions=[".png"],
        id_regex=r"^(.+)\.png$",
    )
    write_crawler_metadata(root, config)

    output = index_dataset_from_path(root, save_index=True)

    assert (root / ".ds_crawler" / "dataset-head.json").is_file()
    assert (root / ".ds_crawler" / "ds-crawler.json").is_file()
    assert (root / ".ds_crawler" / "index.json").is_file()
    assert output["head"]["dataset"]["id"] == "rgb"
    assert output["index"]["files"][0]["id"] == "scene/0001"
