"""Index an augmented-RGB modality (file-id-as-folder) alongside a per-id
GT depth modality so that the resulting trees share hierarchy keys.

Layout::

    rgb/
      abc/aug_1.png  abc/aug_2.png
      xyz/aug_1.png  xyz/aug_2.png
    depth/
      abc.png  xyz.png

After indexing, both ``index["children"]`` mappings are keyed by
``"file_id:abc"`` / ``"file_id:xyz"``.  In ``euler-loading`` you can then
pass depth as a hierarchical modality::

    MultiModalDataset(
        modalities={"rgb": Modality(path=rgb_root, ...)},
        hierarchical_modalities={"depth": Modality(path=depth_root, ...)},
    )

and each ``(file_id, augmentation)`` becomes one sample, with the single
matching depth file shared across all augmentations of the same file-id.

Note: ``ds_crawler.align_datasets`` is *not* the right tool for this
case — it flattens by leaf id only and would collapse augmentations.
The cross-modality join happens at sample-load time inside
``euler-loading`` via the hierarchical-modality prefix walk.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ds_crawler import index_dataset_from_files


RGB_CONFIG = {
    "contract": {"kind": "ds_crawler_config", "version": "2.0"},
    "name": "Augmented RGB",
    "head": {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "augmented_rgb", "name": "Augmented RGB"},
        "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
        "addons": {
            "euler_layout": {
                "version": "1.0",
                "family": "augmented_demo",
                "sample_axis": {"name": "file_id", "location": "hierarchy"},
                "variant_axis": {
                    "name": "augmentation",
                    "location": "file_id",
                },
            },
            "euler_train": {
                "version": "1.0",
                "used_as": "input",
                "slot": "augmented.input.rgb",
            }
        },
    },
    "indexing": {
        "id": {
            # Capture the augmentation name as the file's leaf id.
            "regex": r"^[^/]+/(?P<aug>[^/]+)\.png$",
            "join_char": "+",
        },
        "hierarchy": {
            # Capture the parent folder as the hierarchy key (file-id).
            "regex": r"^(?P<file_id>[^/]+)/[^/]+\.png$",
            "separator": ":",
        },
        "files": {"extensions": [".png"]},
        # Two augs of the same file-id share the same hierarchy node but have
        # distinct ids — not a duplicate event.
        "constraints": {"flat_ids_unique": False},
    },
}


DEPTH_CONFIG = {
    "contract": {"kind": "ds_crawler_config", "version": "2.0"},
    "name": "Per-id Depth",
    "head": {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "per_id_depth", "name": "Per-id Depth"},
        "modality": {
            "key": "depth",
            "meta": {
                "radial_depth": False,
                "scale_to_meters": 1.0,
                "range": [0, 65535],
            },
        },
        "addons": {
            "euler_layout": {
                "version": "1.0",
                "family": "augmented_demo",
                "sample_axis": {"name": "file_id", "location": "hierarchy"},
            },
            "euler_train": {
                "version": "1.0",
                "used_as": "target",
                "slot": "augmented.target.depth",
            }
        },
    },
    "indexing": {
        "id": {
            "regex": r"^(?P<file_id>[^/]+)\.png$",
            "join_char": "+",
        },
        "hierarchy": {
            # Same named group as RGB, so both trees end up keyed by
            # ``file_id:<id>`` at the same depth.
            "regex": r"^(?P<file_id>[^/]+)\.png$",
            "separator": ":",
        },
        "files": {"extensions": [".png"]},
    },
}


def _build_layout(root: Path) -> tuple[Path, Path]:
    rgb_root = root / "rgb"
    depth_root = root / "depth"
    for rel in (
        "abc/aug_1.png", "abc/aug_2.png",
        "xyz/aug_1.png", "xyz/aug_2.png",
    ):
        path = rgb_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    for rel in ("abc.png", "xyz.png"):
        path = depth_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return rgb_root, depth_root


def _summarize_tree(tree: dict) -> dict:
    """Return a compact ``{child_key: [ids]}`` view of a hierarchy node."""
    return {
        child_key: [entry["id"] for entry in child.get("files", [])]
        for child_key, child in tree.get("children", {}).items()
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rgb_root, depth_root = _build_layout(root)

        rgb_files = sorted(rgb_root.rglob("*.png"))
        depth_files = sorted(depth_root.rglob("*.png"))

        rgb_output = index_dataset_from_files(
            RGB_CONFIG, rgb_files, base_path=rgb_root,
        )
        depth_output = index_dataset_from_files(
            DEPTH_CONFIG, depth_files, base_path=depth_root,
        )

        print("RGB tree (augmentations under each file_id):")
        print(json.dumps(_summarize_tree(rgb_output["index"]), indent=2))
        print()
        print("Depth tree (one file under each file_id):")
        print(json.dumps(_summarize_tree(depth_output["index"]), indent=2))
        print()
        rgb_keys = set(rgb_output["index"]["children"].keys())
        depth_keys = set(depth_output["index"]["children"].keys())
        print(f"Shared hierarchy keys: {sorted(rgb_keys & depth_keys)}")


if __name__ == "__main__":
    main()
