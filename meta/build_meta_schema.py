#!/usr/bin/env python3
"""Generate ``schema.json`` from the modality meta schemas defined in config.

Reads ``_MODALITY_META_SCHEMAS`` and writes a JSON Schema file to
``meta/schema.json`` that documents the expected ``properties.meta``
object for each ``euler_train.modality_type``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ds_crawler.config import _MODALITY_META_SCHEMAS

# Map Python types to JSON Schema types.
_TYPE_MAP: dict[type, str] = {
    bool: "boolean",
    int: "number",
    float: "number",
    str: "string",
}


def _json_schema_type(accepted: type | tuple[type, ...]) -> dict:
    """Convert a Python type (or tuple of types) to a JSON Schema type clause."""
    if isinstance(accepted, tuple):
        types = sorted({_TYPE_MAP[t] for t in accepted})
    else:
        types = [_TYPE_MAP[accepted]]

    if len(types) == 1:
        return {"type": types[0]}
    return {"type": types}


def build_schema() -> dict:
    """Build the full JSON Schema from ``_MODALITY_META_SCHEMAS``."""
    modality_schemas: dict[str, dict] = {}

    for modality_type, fields in sorted(_MODALITY_META_SCHEMAS.items()):
        properties: dict[str, dict] = {}
        required: list[str] = []

        for field_name, (accepted_type, _type_label, description) in sorted(fields.items()):
            properties[field_name] = {
                **_json_schema_type(accepted_type),
                "description": description,
            }
            required.append(field_name)

        modality_schemas[modality_type] = {
            "type": "object",
            "description": (
                f"Required meta fields when euler_train.modality_type "
                f"is '{modality_type}'."
            ),
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ds-crawler modality meta schemas",
        "description": (
            "Defines the required properties.meta object for each "
            "euler_train.modality_type value."
        ),
        "type": "object",
        "properties": modality_schemas,
    }


def main() -> None:
    schema = build_schema()
    out_path = Path(__file__).resolve().parent / "schema.json"
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
