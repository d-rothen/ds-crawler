"""Tests for the generated UI-oriented meta JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from meta.build_meta_schema import build_schema


class TestMetaSchema:
    def test_checked_in_schema_is_current(self) -> None:
        schema_path = Path(__file__).parents[1] / "meta" / "schema.json"

        assert json.loads(schema_path.read_text()) == build_schema()

    def test_dimensions_schema_supports_dynamic_form_hints(self) -> None:
        schema = build_schema()
        dimensions = schema["properties"]["rgb"]["properties"]["dimensions"]

        assert dimensions["type"] == "object"
        assert dimensions["propertyNames"] == {
            "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
        }
        assert dimensions["additionalProperties"] == {
            "type": "integer",
            "minimum": 1,
        }
        assert dimensions["x-ui"] == {
            "widget": "keyValueTable",
            "keyLabel": "Axis",
            "valueLabel": "Size",
            "allowCustomKeys": True,
            "suggestedKeys": [
                "height",
                "width",
                "channels",
                "depth",
                "time",
                "features",
            ],
        }

    def test_depth_range_schema_is_explicit_two_number_array(self) -> None:
        schema = build_schema()
        depth_range = schema["properties"]["depth"]["properties"]["range"]

        assert depth_range["type"] == "array"
        assert depth_range["items"] == {"type": "number"}
        assert depth_range["minItems"] == 2
        assert depth_range["maxItems"] == 2
