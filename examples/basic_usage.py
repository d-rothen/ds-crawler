"""Build canonical artifacts for a tiny temporary RGB dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ds_crawler import build_dataset_artifacts_from_files


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        files = [
            root / "scene_01" / "0001.png",
            root / "scene_01" / "0002.png",
            root / "scene_02" / "0001.png",
        ]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"example")

        result = build_dataset_artifacts_from_files(
            dataset={"id": "demo_rgb", "name": "Demo RGB"},
            modality={"key": "rgb", "meta": {"range": [0, 255]}},
            indexing={
                "id": {
                    "regex": r"^[^/]+/(?P<frame>\d+)\.png$",
                    "join_char": "+",
                },
                "hierarchy": {
                    "regex": r"^(?P<scene>[^/]+)/",
                    "separator": ":",
                },
                "files": {"extensions": [".png"]},
                "constraints": {"flat_ids_unique": False},
            },
            files=files,
            base_path=root,
        )

        print(json.dumps(result["summary"], indent=2))
        print("Artifacts:", ", ".join(result["artifacts"]))


if __name__ == "__main__":
    main()
