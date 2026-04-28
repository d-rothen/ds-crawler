from __future__ import annotations

from pathlib import Path
from typing import Any

from ds_crawler import (
    DatasetWriter,
    copy_dataset,
    create_dataset_splits,
    index_dataset_from_files,
    index_dataset_from_path,
)
from ds_crawler.validation import validate_output

from .current_helpers import create_files, sample_head


def test_end_to_end_directory_to_zip_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    writer = DatasetWriter(source, head=sample_head(name="RGB"))
    writer.get_path("/scene/0001", "0001.png").write_bytes(b"data")
    writer.get_path("/scene/0002", "0002.png").write_bytes(b"data")
    writer.save_index()
    create_dataset_splits(source, ["train"], [1.0], seed=4)

    target_zip = tmp_path / "dataset.zip"
    copy_dataset(source, target_zip)
    loaded = index_dataset_from_path(target_zip)

    assert loaded["head"]["dataset"]["name"] == "RGB"
    assert loaded["head"]["modality"]["meta"]["file_types"] == ["png"]
    assert validate_output(loaded)["contract"]["kind"] == "dataset_index"


# ---------------------------------------------------------------------------
# Augmented modality (file-id-as-folder) matched against per-id GT
# ---------------------------------------------------------------------------
#
# Layout under test:
#
#   rgb/abc/aug_1.png    rgb/abc/aug_2.png
#   rgb/xyz/aug_1.png    rgb/xyz/aug_2.png
#   depth/abc.png        depth/xyz.png
#
# Recipe: both indices place files under matching ``children["file_id:<id>"]``
# nodes so that euler-loading's ``hierarchical_modalities=`` can join
# per-augmentation RGB samples to the single GT depth file at the same
# hierarchy prefix.


def _augmented_rgb_config(root: Path) -> dict[str, Any]:
    return {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "name": "Augmented RGB",
        "path": str(root),
        "head": sample_head(name="Augmented RGB", modality="rgb"),
        "source": {"path": str(root)},
        "indexing": {
            "id": {
                "regex": r"^[^/]+/(?P<aug>[^/]+)\.png$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": r"^(?P<file_id>[^/]+)/[^/]+\.png$",
                "separator": ":",
            },
            "files": {"extensions": [".png"]},
            "constraints": {"flat_ids_unique": False},
        },
    }


def _per_id_depth_config(root: Path) -> dict[str, Any]:
    return {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "name": "Per-id Depth",
        "path": str(root),
        "head": sample_head(name="Per-id Depth", modality="depth"),
        "source": {"path": str(root)},
        "indexing": {
            "id": {
                "regex": r"^(?P<file_id>[^/]+)\.png$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": r"^(?P<file_id>[^/]+)\.png$",
                "separator": ":",
            },
            "files": {"extensions": [".png"]},
            "constraints": {"flat_ids_unique": True},
        },
    }


class TestAugmentedRgbWithSharedDepth:
    """Verify that ds-crawler can index a file-id-as-folder augmented modality
    in a shape that lines up with a per-id GT depth modality.

    The hypothesis is that no production code changes are needed: configuring
    each modality's ``hierarchy.regex`` to extract the same ``file_id`` named
    group makes both indices share child keys (``"file_id:abc"``,
    ``"file_id:xyz"``) at the same depth.
    """

    def _build_layout(self, tmp_path: Path) -> tuple[Path, Path, list[Path], list[Path]]:
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        rgb_files = create_files(
            rgb_root,
            [
                "abc/aug_1.png",
                "abc/aug_2.png",
                "xyz/aug_1.png",
                "xyz/aug_2.png",
            ],
        )
        depth_files = create_files(depth_root, ["abc.png", "xyz.png"])
        return rgb_root, depth_root, rgb_files, depth_files

    def test_rgb_index_groups_augs_under_file_id_child(self, tmp_path: Path) -> None:
        rgb_root, _depth_root, rgb_files, _depth_files = self._build_layout(tmp_path)

        output = index_dataset_from_files(
            _augmented_rgb_config(rgb_root), rgb_files, base_path=rgb_root,
        )

        children = output["index"]["children"]
        assert set(children.keys()) == {"file_id:abc", "file_id:xyz"}
        assert output["index"].get("files", []) == []  # nothing at the root

        for file_id_key in ("file_id:abc", "file_id:xyz"):
            entries = children[file_id_key]["files"]
            assert len(entries) == 2
            ids = sorted(entry["id"] for entry in entries)
            assert ids == ["aug-aug_1", "aug-aug_2"]
            # Children carry no nested children.
            assert children[file_id_key].get("children", {}) == {}

    def test_depth_index_places_file_under_matching_file_id_child(
        self, tmp_path: Path,
    ) -> None:
        _rgb_root, depth_root, _rgb_files, depth_files = self._build_layout(tmp_path)

        output = index_dataset_from_files(
            _per_id_depth_config(depth_root), depth_files, base_path=depth_root,
        )

        children = output["index"]["children"]
        assert set(children.keys()) == {"file_id:abc", "file_id:xyz"}
        assert output["index"].get("files", []) == []

        for file_id_key, expected_path in (
            ("file_id:abc", "abc.png"),
            ("file_id:xyz", "xyz.png"),
        ):
            entries = children[file_id_key]["files"]
            assert len(entries) == 1
            assert entries[0]["path"] == expected_path
            assert entries[0]["id"] == file_id_key.replace(":", "-")

    def test_rgb_and_depth_share_hierarchy_keys(self, tmp_path: Path) -> None:
        """The actual cross-modality join precondition: both indices use the
        same ``children`` keys at the file-id level so a hierarchical
        modality lookup finds the depth file by hierarchy prefix.
        """
        rgb_root, depth_root, rgb_files, depth_files = self._build_layout(tmp_path)

        rgb_output = index_dataset_from_files(
            _augmented_rgb_config(rgb_root), rgb_files, base_path=rgb_root,
        )
        depth_output = index_dataset_from_files(
            _per_id_depth_config(depth_root), depth_files, base_path=depth_root,
        )

        assert (
            set(rgb_output["index"]["children"].keys())
            == set(depth_output["index"]["children"].keys())
        )

    def test_strict_indexing_does_not_flag_augs_as_duplicates(
        self, tmp_path: Path,
    ) -> None:
        """With ``flat_ids_unique=False``, two augs of the same file-id at the
        same hierarchy level are *not* duplicates because their ids
        (``aug-aug_1`` vs ``aug-aug_2``) differ.
        """
        rgb_root, _depth_root, rgb_files, _depth_files = self._build_layout(tmp_path)

        # strict=True would raise on a duplicate id event.
        output = index_dataset_from_files(
            _augmented_rgb_config(rgb_root),
            rgb_files,
            base_path=rgb_root,
            strict=True,
        )

        total = sum(
            len(child["files"]) for child in output["index"]["children"].values()
        )
        assert total == 4
