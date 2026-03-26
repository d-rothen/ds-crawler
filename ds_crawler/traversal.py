"""Hierarchy traversal, collection, and filtering utilities.

Pure functions that walk the nested ``children`` / ``files`` structure
produced by :class:`~ds_crawler.parser.DatasetParser` and returned in
``output.json`` dicts.  These functions have no dependencies on the
rest of the package — they operate on plain dicts.
"""

from __future__ import annotations

import math
import random
from numbers import Real
from typing import Any


_RATIO_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Collect: extract data from the hierarchy
# ---------------------------------------------------------------------------


def get_files(output_json: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    """Return a flat list of file paths from an output JSON.

    Accepts either a single dataset output dict (as returned by
    ``index_dataset``) or a list of them (the full output JSON file).

    Args:
        output_json: A dataset output dict or a list of dataset output dicts.

    Returns:
        A flat list of every file path found in the output.
    """
    datasets = output_json if isinstance(output_json, list) else [output_json]
    paths: list[str] = []
    for dataset in datasets:
        _collect_paths(dataset.get("dataset", {}), paths)
    return paths


def _collect_paths(node: dict[str, Any], paths: list[str]) -> None:
    """Recursively collect file paths from a hierarchy node."""
    for file_entry in node.get("files", []):
        path = file_entry.get("path")
        if path is not None:
            paths.append(path)
    for child in node.get("children", {}).values():
        _collect_paths(child, paths)


def _collect_ids(output_json: dict[str, Any]) -> set[str]:
    """Extract all file IDs from an output JSON dict's hierarchy.

    Recursively walks the ``dataset`` node and collects the ``id`` field
    from every file entry.

    Args:
        output_json: A single dataset output dict (as returned by
            ``index_dataset``).

    Returns:
        A set of all file ID strings found in the hierarchy.
    """
    ids: set[str] = set()
    _collect_ids_from_node(output_json.get("dataset", {}), ids)
    return ids


def _collect_ids_from_node(node: dict[str, Any], ids: set[str]) -> None:
    """Recursively collect file IDs from a hierarchy node."""
    for file_entry in node.get("files", []):
        file_id = file_entry.get("id")
        if file_id is not None:
            ids.add(file_id)
    for child in node.get("children", {}).values():
        _collect_ids_from_node(child, ids)


def _collect_qualified_ids(output_json: dict[str, Any]) -> set[tuple[str, ...]]:
    """Extract hierarchy-qualified file IDs from an output JSON dict.

    Each ID is a tuple of ``(*hierarchy_keys, file_id)`` so that files
    with identical IDs under different hierarchy levels are distinguished.

    Args:
        output_json: A single dataset output dict (as returned by
            ``index_dataset``).

    Returns:
        A set of tuples, each containing the hierarchy path keys
        followed by the file ID.
    """
    ids: set[tuple[str, ...]] = set()
    _collect_qualified_ids_from_node(output_json.get("dataset", {}), (), ids)
    return ids


def _collect_qualified_ids_from_node(
    node: dict[str, Any],
    path: tuple[str, ...],
    ids: set[tuple[str, ...]],
) -> None:
    """Recursively collect hierarchy-qualified IDs from a node."""
    for file_entry in node.get("files", []):
        file_id = file_entry.get("id")
        if file_id is not None:
            ids.add(path + (file_id,))
    for name, child in node.get("children", {}).items():
        _collect_qualified_ids_from_node(child, path + (name,), ids)


def collect_qualified_ids(output_json: dict[str, Any]) -> set[tuple[str, ...]]:
    """Extract hierarchy-qualified file IDs from an output JSON dict.

    Each ID is a tuple of ``(*hierarchy_keys, file_id)`` so that files
    with identical IDs under different hierarchy levels are distinguished.

    Args:
        output_json: A single dataset output dict (as returned by
            ``index_dataset``).

    Returns:
        A set of tuples, each containing the hierarchy path keys
        followed by the file ID.
    """
    return _collect_qualified_ids(output_json)


def _collect_all_referenced_paths(output_json: dict[str, Any]) -> list[str]:
    """Collect ALL file paths referenced in an output JSON dict.

    This includes:

    - File paths from file entries (the ``path`` field)
    - ``camera_intrinsics`` paths
    - ``camera_extrinsics`` paths

    Args:
        output_json: A single dataset output dict.

    Returns:
        A list of all referenced relative file paths (may contain
        duplicates when the same camera file is shared across hierarchy
        levels).
    """
    paths: list[str] = []
    _collect_all_paths_from_node(output_json.get("dataset", {}), paths)
    return paths


def _collect_all_paths_from_node(
    node: dict[str, Any], paths: list[str]
) -> None:
    """Recursively collect all referenced paths from a hierarchy node."""
    for file_entry in node.get("files", []):
        path = file_entry.get("path")
        if path is not None:
            paths.append(path)
    for key in ("camera_intrinsics", "camera_extrinsics"):
        cam_path = node.get(key)
        if cam_path is not None:
            paths.append(cam_path)
    for child in node.get("children", {}).values():
        _collect_all_paths_from_node(child, paths)


def _collect_file_entries_by_id(
    node: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Recursively collect ``{id: file_entry}`` from a hierarchy node."""
    entries: dict[str, dict[str, Any]] = {}
    for file_entry in node.get("files", []):
        file_id = file_entry.get("id")
        if file_id is not None:
            entries[file_id] = file_entry
    for child in node.get("children", {}).values():
        entries.update(_collect_file_entries_by_id(child))
    return entries


# ---------------------------------------------------------------------------
# Filter: narrow down an index to a subset
# ---------------------------------------------------------------------------


def _filter_index_by_paths(
    output_json: dict[str, Any], keep_paths: set[str]
) -> dict[str, Any]:
    """Return a shallow copy of *output_json* with file entries filtered.

    Only file entries whose ``path`` is in *keep_paths* are retained.
    Camera intrinsics/extrinsics and all other metadata are preserved.
    """
    result = dict(output_json)
    if "dataset" in result:
        result["dataset"] = _filter_node_by_paths(result["dataset"], keep_paths)
    return result


def _filter_node_by_paths(
    node: dict[str, Any], keep_paths: set[str]
) -> dict[str, Any]:
    """Recursively filter file entries in a hierarchy node."""
    filtered: dict[str, Any] = {}
    for key, value in node.items():
        if key == "files":
            filtered[key] = [f for f in value if f.get("path") in keep_paths]
        elif key == "children":
            filtered[key] = {
                name: _filter_node_by_paths(child, keep_paths)
                for name, child in value.items()
            }
        else:
            filtered[key] = value
    return filtered


def filter_index_by_qualified_ids(
    output_json: dict[str, Any],
    qualified_ids: set[tuple[str, ...]],
) -> dict[str, Any]:
    """Return a copy of *output_json* keeping only entries matching *qualified_ids*.

    Each qualified ID is a tuple of ``(*hierarchy_keys, file_id)``.  Only
    file entries whose hierarchy path + id match an entry in *qualified_ids*
    are retained.  Camera intrinsics/extrinsics and other node-level
    metadata are preserved.

    Args:
        output_json: A single dataset output dict.
        qualified_ids: Set of ``(*hierarchy_keys, file_id)`` tuples to keep.

    Returns:
        A filtered copy of the output dict.
    """
    result = dict(output_json)
    if "dataset" in result:
        result["dataset"] = _filter_node_by_qualified_ids(
            result["dataset"], (), qualified_ids
        )
    return result


def _filter_node_by_qualified_ids(
    node: dict[str, Any],
    path: tuple[str, ...],
    qualified_ids: set[tuple[str, ...]],
) -> dict[str, Any]:
    """Recursively filter file entries by hierarchy-qualified IDs."""
    filtered: dict[str, Any] = {}
    for key, value in node.items():
        if key == "files":
            filtered[key] = [
                f for f in value
                if path + (f.get("id"),) in qualified_ids
            ]
        elif key == "children":
            filtered[key] = {
                name: _filter_node_by_qualified_ids(
                    child, path + (name,), qualified_ids
                )
                for name, child in value.items()
            }
        else:
            filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# Split: partition qualified IDs by ratio
# ---------------------------------------------------------------------------


def _validate_split_sample(sample: int | None) -> int | None:
    """Validate optional split sampling stride."""
    if sample is None:
        return None
    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        raise ValueError(f"sample must be a positive integer, got {sample!r}")
    return sample


def _normalize_split_ratios(ratios: list[int | float]) -> list[float]:
    """Normalize ratios to fractions of the selected candidate pool."""
    if not ratios:
        raise ValueError("ratios must be non-empty")

    values: list[float] = []
    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, Real):
            raise ValueError(f"All ratios must be finite numbers, got {ratios}")
        value = float(ratio)
        if not math.isfinite(value):
            raise ValueError(f"All ratios must be finite numbers, got {ratios}")
        if value <= 0:
            raise ValueError(f"All ratios must be positive, got {ratios}")
        values.append(value)

    total = sum(values)
    if all(value <= 1.0 + _RATIO_EPSILON for value in values):
        if total > 1.0 + _RATIO_EPSILON:
            raise ValueError(
                f"Fractional ratios must sum to <= 1, got {total} from {ratios}"
            )
        return values

    if total > 100.0 + _RATIO_EPSILON:
        raise ValueError(
            f"Percentage ratios must sum to <= 100, got {total} from {ratios}"
        )
    return [value / 100.0 for value in values]


def _prepare_split_candidates(
    qualified_ids: set[tuple[str, ...]],
    *,
    sample: int | None = None,
    seed: int | None = None,
) -> list[tuple[str, ...]]:
    """Return ordered candidates after optional stride sampling and shuffling."""
    validated_sample = _validate_split_sample(sample)

    ordered = sorted(qualified_ids)
    if validated_sample is not None and validated_sample > 1:
        ordered = ordered[::validated_sample]

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    return ordered


def split_qualified_ids(
    qualified_ids: set[tuple[str, ...]],
    ratios: list[int | float],
    *,
    seed: int | None = None,
    sample: int | None = None,
) -> list[set[tuple[str, ...]]]:
    """Split qualified IDs into groups according to numeric ratios.

    Args:
        qualified_ids: Set of ``(*hierarchy_keys, file_id)`` tuples.
        ratios: Positive percentages (e.g. ``[80, 20]``) or fractions
            (e.g. ``[0.8, 0.2]``). Totals may be less than full coverage,
            in which case some sampled IDs remain unassigned.
        seed: Random seed for shuffling before splitting.  When ``None``
            the IDs are split in sorted order (deterministic but not
            randomised).
        sample: Optional stride applied before splitting. ``sample=5``
            keeps every fifth qualified ID from the sorted candidate pool,
            then applies the split ratios to that sampled subset.

    Returns:
        A list of sets (one per ratio) partitioning the sampled candidate
        pool. IDs not covered by the ratios are left out of every split.

    Raises:
        ValueError: If *ratios* are invalid or *sample* is not a positive
            integer.
    """
    fractions = _normalize_split_ratios(ratios)
    ordered = _prepare_split_candidates(
        qualified_ids,
        sample=sample,
        seed=seed,
    )

    total = len(ordered)
    boundaries: list[int] = []
    cumsum = 0.0
    for fraction in fractions:
        cumsum += fraction
        if math.isclose(cumsum, 1.0, rel_tol=0.0, abs_tol=_RATIO_EPSILON):
            boundary = total
        else:
            boundary = round(total * cumsum)
        boundaries.append(min(boundary, total))

    splits: list[set[tuple[str, ...]]] = []
    start = 0
    for end in boundaries:
        splits.append(set(ordered[start:end]))
        start = end

    return splits
