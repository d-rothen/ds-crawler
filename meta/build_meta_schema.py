#!/usr/bin/env python3
"""Generate ``schema.json`` from the shared euler-modalities registry."""

from __future__ import annotations

import json
from pathlib import Path

from ds_crawler._euler_modalities import build_meta_schema


def build_schema() -> dict:
    """Build the full JSON Schema from the shared modality registry."""
    return build_meta_schema()


def main() -> None:
    schema = build_schema()
    out_path = Path(__file__).resolve().parent / "schema.json"
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
