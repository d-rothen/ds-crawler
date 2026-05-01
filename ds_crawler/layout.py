"""Helpers for Euler dataset layout metadata.

The ``euler_layout`` addon describes how files in a modality relate to the
logical sample axis and optional variant/augmentation axis.  It is intentionally
small and namespaced so dataset-head contracts remain backwards compatible.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


EULER_LAYOUT_ADDON = "euler_layout"
EULER_LAYOUT_VERSION = "1.0"
AXIS_LOCATIONS = {"file_id", "hierarchy", "attributes"}


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, context)


def _axis(
    *,
    name: str,
    location: str,
    key: str | None = None,
) -> dict[str, str]:
    result = {"name": name, "location": location}
    if key is not None:
        result["key"] = key
    return result


def build_layout_addon(
    *,
    family: str | None = None,
    sample_axis_name: str = "file_id",
    sample_axis_location: str = "file_id",
    sample_axis_key: str | None = None,
    variant_axis_name: str | None = None,
    variant_axis_location: str = "file_id",
    variant_axis_key: str | None = None,
    derived_from: Mapping[str, Any] | None = None,
    version: str = EULER_LAYOUT_VERSION,
) -> dict[str, Any]:
    """Build and validate an ``euler_layout`` addon payload."""
    payload: dict[str, Any] = {
        "version": version,
        "sample_axis": _axis(
            name=sample_axis_name,
            location=sample_axis_location,
            key=sample_axis_key,
        ),
    }
    if family is not None:
        payload["family"] = family
    if variant_axis_name is not None:
        payload["variant_axis"] = _axis(
            name=variant_axis_name,
            location=variant_axis_location,
            key=variant_axis_key,
        )
    if derived_from is not None:
        payload["derived_from"] = dict(derived_from)
    return validate_layout_addon(payload)


def validate_layout_addon(
    payload: Mapping[str, Any],
    *,
    context: str = EULER_LAYOUT_ADDON,
) -> dict[str, Any]:
    """Validate and normalize an ``euler_layout`` addon payload."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")

    result = deepcopy(dict(payload))
    result["version"] = _non_empty_string(
        result.get("version"), f"{context}.version"
    )
    family = _optional_string(result.get("family"), f"{context}.family")
    if family is None:
        result.pop("family", None)

    sample_axis = result.get("sample_axis")
    if not isinstance(sample_axis, Mapping):
        raise ValueError(f"{context}.sample_axis must be an object")
    result["sample_axis"] = _validate_axis(
        sample_axis, context=f"{context}.sample_axis"
    )

    variant_axis = result.get("variant_axis")
    if variant_axis is None:
        result.pop("variant_axis", None)
    else:
        if not isinstance(variant_axis, Mapping):
            raise ValueError(f"{context}.variant_axis must be an object")
        result["variant_axis"] = _validate_axis(
            variant_axis, context=f"{context}.variant_axis"
        )

    derived_from = result.get("derived_from")
    if derived_from is None:
        result.pop("derived_from", None)
    elif not isinstance(derived_from, Mapping):
        raise ValueError(f"{context}.derived_from must be an object")
    else:
        result["derived_from"] = dict(derived_from)

    return result


def _validate_axis(value: Mapping[str, Any], *, context: str) -> dict[str, str]:
    axis = dict(value)
    name = _non_empty_string(axis.get("name"), f"{context}.name")
    location = _non_empty_string(axis.get("location"), f"{context}.location")
    if location not in AXIS_LOCATIONS:
        allowed = ", ".join(sorted(AXIS_LOCATIONS))
        raise ValueError(f"{context}.location must be one of: {allowed}")

    result = {"name": name, "location": location}
    key = _optional_string(axis.get("key"), f"{context}.key")
    if key is not None:
        result["key"] = key
    return result


def get_layout_addon(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a validated ``euler_layout`` addon from a head or index mapping."""
    head = value.get("head")
    if isinstance(head, Mapping):
        addons = head.get("addons")
        if isinstance(addons, Mapping):
            layout = addons.get(EULER_LAYOUT_ADDON)
            if isinstance(layout, Mapping):
                return validate_layout_addon(layout)

    addons = value.get("addons")
    if isinstance(addons, Mapping):
        layout = addons.get(EULER_LAYOUT_ADDON)
        if isinstance(layout, Mapping):
            return validate_layout_addon(layout)

    layout = value.get(EULER_LAYOUT_ADDON)
    if isinstance(layout, Mapping):
        return validate_layout_addon(layout)
    return None


__all__ = [
    "AXIS_LOCATIONS",
    "EULER_LAYOUT_ADDON",
    "EULER_LAYOUT_VERSION",
    "build_layout_addon",
    "get_layout_addon",
    "validate_layout_addon",
]
