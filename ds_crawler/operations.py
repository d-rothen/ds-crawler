"""High-level dataset operations: align, copy, split.

These functions compose the traversal utilities from
:mod:`~ds_crawler.traversal` with the indexing API from
:mod:`~ds_crawler.parser` to provide bulk dataset manipulation.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    build_crawler_config_for_output,
    build_index_artifact,
    build_split_artifact,
    hydrate_split_artifact,
    save_output_artifacts,
)
from .config import CONFIG_FILENAME
from .traversal import (
    _collect_all_referenced_paths,
    _collect_file_entries_by_id,
    _collect_qualified_ids,
    _filter_index_by_paths,
    _get_index_node,
    _prepare_split_candidates,
    filter_index_by_qualified_ids,
    get_files,
    split_qualified_ids,
)
from .zip_utils import (
    COMPRESSED_EXTENSIONS,
    DATASET_HEAD_FILENAME,
    METADATA_DIR,
    OUTPUT_FILENAME,
    get_metadata_entry_name,
    get_split_filename,
    is_zip_path,
    list_split_names,
    read_metadata_json,
    validate_split_name,
    write_metadata_json_batch,
    write_metadata_json,
)
from .validation import validate_split_artifact

logger = logging.getLogger(__name__)

# Backwards-compatible alias
_COMPRESSED_EXTENSIONS = COMPRESSED_EXTENSIONS


# ---------------------------------------------------------------------------
# Inline splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchySplitClause:
    """One hierarchy-level predicate for a hierarchy-based inline split."""

    level_index: int
    values: tuple[str, ...]

    def to_execution_dict(self) -> dict[str, Any]:
        return {
            "levelIndex": self.level_index,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class HierarchySplitRule:
    """Named hierarchy rule used to create one inline split artifact."""

    name: str
    clauses: tuple[HierarchySplitClause, ...]

    def clauses_for_execution(self) -> list[dict[str, Any]]:
        return [clause.to_execution_dict() for clause in self.clauses]


def _normalize_split_names(split_names: list[str]) -> list[str]:
    """Validate split names and reject duplicates."""
    normalized = [validate_split_name(name) for name in split_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("split_names must be unique")
    return normalized


def _normalize_hierarchy_split_rules(
    hierarchy_rules: dict[str, Any] | list[dict[str, Any]],
    *,
    exclusive: bool | None = None,
) -> tuple[bool, list[HierarchySplitRule]]:
    """Validate and normalize hierarchy split rule input.

    Accepts either the Euler View request shape
    ``{"exclusive": bool, "splits": [...]}`` or a bare list of split rules.
    Clause level indices are zero-based hierarchy depths, and values are
    matched against the exact child keys stored in ``index.children``.
    """
    if exclusive is not None and not isinstance(exclusive, bool):
        raise ValueError("exclusive must be a bool when provided")

    if isinstance(hierarchy_rules, dict):
        raw_splits = hierarchy_rules.get("splits")
        raw_exclusive = hierarchy_rules.get("exclusive", True)
        if raw_exclusive is not None and not isinstance(raw_exclusive, bool):
            raise ValueError("hierarchy_rules.exclusive must be a bool")
        resolved_exclusive = raw_exclusive is not False
    elif isinstance(hierarchy_rules, list):
        raw_splits = hierarchy_rules
        resolved_exclusive = True
    else:
        raise ValueError("hierarchy_rules must be an object or list")

    if exclusive is not None:
        resolved_exclusive = exclusive

    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError("hierarchy_rules must include at least one split rule")

    rules: list[HierarchySplitRule] = []
    for split_index, raw_split in enumerate(raw_splits):
        if not isinstance(raw_split, dict):
            raise ValueError(f"Hierarchy split rule {split_index} must be an object")
        name = validate_split_name(raw_split.get("name"))
        raw_clauses = raw_split.get("clauses")
        if not isinstance(raw_clauses, list) or not raw_clauses:
            raise ValueError(
                f"Hierarchy split {name!r} must include at least one clause"
            )

        clauses: list[HierarchySplitClause] = []
        seen_levels: set[int] = set()
        for clause_index, raw_clause in enumerate(raw_clauses):
            if not isinstance(raw_clause, dict):
                raise ValueError(
                    f"Hierarchy split {name!r} clause {clause_index} must be an object"
                )

            raw_level = raw_clause.get("levelIndex", raw_clause.get("level_index"))
            if (
                isinstance(raw_level, bool)
                or not isinstance(raw_level, int)
                or raw_level < 0
            ):
                raise ValueError(
                    f"Hierarchy split {name!r} has an invalid hierarchy level"
                )
            if raw_level in seen_levels:
                raise ValueError(
                    f"Hierarchy split {name!r} has duplicate rules for level {raw_level}"
                )
            seen_levels.add(raw_level)

            raw_values = raw_clause.get("values")
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(
                    f"Hierarchy split {name!r} level {raw_level} must include values"
                )
            values: list[str] = []
            for value in raw_values:
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"Hierarchy split {name!r} level {raw_level} values "
                        "must be non-empty strings"
                    )
                if value not in values:
                    values.append(value)
            clauses.append(
                HierarchySplitClause(level_index=raw_level, values=tuple(values))
            )

        rules.append(HierarchySplitRule(name=name, clauses=tuple(clauses)))

    normalized_names = _normalize_split_names([rule.name for rule in rules])
    if normalized_names != [rule.name for rule in rules]:
        rules = [
            HierarchySplitRule(name=name, clauses=rule.clauses)
            for name, rule in zip(normalized_names, rules)
        ]
    return resolved_exclusive, rules


def _qualified_id_matches_hierarchy_rule(
    qualified_id: tuple[str, ...],
    rule: HierarchySplitRule,
) -> bool:
    hierarchy_path = qualified_id[:-1]
    for clause in rule.clauses:
        if clause.level_index >= len(hierarchy_path):
            return False
        if hierarchy_path[clause.level_index] not in clause.values:
            return False
    return True


def _select_hierarchy_rule_splits(
    qualified_ids: set[tuple[str, ...]],
    rules: list[HierarchySplitRule],
    *,
    exclusive: bool,
    sample: int | None,
) -> tuple[list[set[tuple[str, ...]]], list[int]]:
    """Resolve hierarchy rules to qualified-ID split sets."""
    assigned_before_sampling: set[tuple[str, ...]] = set()
    id_splits: list[set[tuple[str, ...]]] = []
    matched_counts: list[int] = []

    for rule in rules:
        matched_ids = {
            qualified_id
            for qualified_id in qualified_ids
            if _qualified_id_matches_hierarchy_rule(qualified_id, rule)
        }
        effective_ids = (
            matched_ids - assigned_before_sampling
            if exclusive else matched_ids
        )
        selected_ids = set(_prepare_split_candidates(effective_ids, sample=sample))
        if not selected_ids:
            raise ValueError(
                f"Hierarchy split {rule.name!r} has no files after "
                "exclusivity and sampling"
            )
        id_splits.append(selected_ids)
        matched_counts.append(len(effective_ids))
        if exclusive:
            assigned_before_sampling.update(effective_ids)

    return id_splits, matched_counts


def _split_metadata_path(
    dataset_path: Path,
    filename: str,
    *,
    metadata_scope: str | None = None,
) -> str:
    if is_zip_path(dataset_path):
        return str(dataset_path)
    if metadata_scope is not None:
        return str(dataset_path / METADATA_DIR / metadata_scope / filename)
    return str(dataset_path / METADATA_DIR / filename)


def _ensure_output_index(
    dataset_path: Path,
    index: dict[str, Any] | None = None,
    *,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Return a full output index, indexing and saving it when needed."""
    if index is not None:
        if read_metadata_json(
            dataset_path,
            OUTPUT_FILENAME,
            metadata_scope=metadata_scope,
        ) is None:
            save_output_artifacts(
                dataset_path,
                index,
                metadata_scope=metadata_scope,
            )
        return index

    from .parser import index_dataset_from_path
    cached = read_metadata_json(
        dataset_path,
        OUTPUT_FILENAME,
        metadata_scope=metadata_scope,
    )
    if cached is not None:
        return index_dataset_from_path(
            dataset_path,
            metadata_scope=metadata_scope,
        )

    logger.info(
        "No %s found at %s, indexing dataset before writing splits",
        OUTPUT_FILENAME,
        dataset_path,
    )
    return index_dataset_from_path(
        dataset_path,
        save_index=True,
        metadata_scope=metadata_scope,
    )


def _build_split_index_payload(
    dataset_path: Path,
    index: dict[str, Any],
    split_name: str,
    split_ids: set[tuple[str, ...]],
    *,
    ratio: int | float | None = None,
    seed: int | None = None,
    sample: int | None = None,
    execution: dict[str, Any] | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Build a split artifact and result summary without writing it."""
    filename = get_split_filename(split_name)
    filtered_index = filter_index_by_qualified_ids(index, split_ids)
    split_execution: dict[str, Any] = dict(execution or {})
    if ratio is not None:
        split_execution["ratio"] = ratio
    if seed is not None:
        split_execution["seed"] = seed
    if sample is not None:
        split_execution["sampled"] = sample
    artifact = build_split_artifact(
        filtered_index,
        split_name=split_name,
        execution=split_execution or None,
    )
    result: dict[str, Any] = {
        "split": split_name,
        "filename": filename,
        "metadata_file": get_metadata_entry_name(
            filename,
            metadata_scope=metadata_scope,
        ),
        "path": _split_metadata_path(
            dataset_path,
            filename,
            metadata_scope=metadata_scope,
        ),
        "num_ids": len(split_ids),
        "artifact": artifact,
    }
    if ratio is not None:
        result["ratio"] = ratio
    return result


def _write_split_index(
    dataset_path: Path,
    index: dict[str, Any],
    split_name: str,
    split_ids: set[tuple[str, ...]],
    *,
    ratio: int | float | None = None,
    seed: int | None = None,
    sample: int | None = None,
    execution: dict[str, Any] | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Persist a single split dataset artifact under ``.ds_crawler/``."""
    payload = _build_split_index_payload(
        dataset_path,
        index,
        split_name,
        split_ids,
        ratio=ratio,
        seed=seed,
        sample=sample,
        execution=execution,
        metadata_scope=metadata_scope,
    )
    write_metadata_json(
        dataset_path,
        payload["filename"],
        payload["artifact"],
        metadata_scope=metadata_scope,
    )
    result = dict(payload)
    result.pop("artifact")
    return result


def _write_split_payloads_batch(
    dataset_path: Path,
    payloads: list[dict[str, Any]],
    *,
    metadata_scope: str | None = None,
) -> list[dict[str, Any]]:
    """Write split artifacts in one metadata pass and return public summaries."""
    if not payloads:
        return []
    write_metadata_json_batch(
        dataset_path,
        {payload["filename"]: payload["artifact"] for payload in payloads},
        metadata_scope=metadata_scope,
    )
    results: list[dict[str, Any]] = []
    for payload in payloads:
        result = dict(payload)
        result.pop("artifact")
        results.append(result)
    return results


def list_dataset_splits(
    dataset_path: str | Path,
    *,
    metadata_scope: str | None = None,
) -> list[str]:
    """Return sorted split names available for a dataset."""
    return list_split_names(Path(dataset_path), metadata_scope=metadata_scope)


def load_dataset_split(
    dataset_path: str | Path,
    split_name: str,
    *,
    strict: bool = False,
    save_index: bool = False,
    force_reindex: bool = False,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Load a named split as a full output dict.

    The returned object has the same top-level metadata as ``index.json``,
    but its ``index`` payload is replaced by the split artifact's index
    content and includes the split descriptor under ``result["split"]``.
    """
    dataset_path = Path(dataset_path)

    from .parser import index_dataset_from_path

    base_output = index_dataset_from_path(
        dataset_path,
        strict=strict,
        save_index=save_index,
        force_reindex=force_reindex,
        metadata_scope=metadata_scope,
    )
    split_artifact = _load_required_index(
        dataset_path,
        get_split_filename(split_name),
        metadata_scope=metadata_scope,
    )
    validate_split_artifact(split_artifact, context=f"split[{split_name!r}]")

    result = hydrate_split_artifact(split_artifact, base_output)
    result.pop("dataset", None)
    return result


def create_dataset_splits(
    source_path: str | Path,
    split_names: list[str],
    ratios: list[int | float],
    *,
    index: dict[str, Any] | None = None,
    qualified_ids: set[tuple[str, ...]] | None = None,
    seed: int | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Create inline split metadata files for a single dataset.

    The full index remains stored as ``index.json``. Each split is written
    as ``.ds_crawler/split_<name>.json`` and contains a split artifact
    with contract metadata, provenance, and the filtered ``index`` node.
    """
    if len(split_names) != len(ratios):
        raise ValueError(
            f"split_names and ratios must have the same length, "
            f"got {len(split_names)} and {len(ratios)}"
        )

    normalized_names = _normalize_split_names(split_names)
    source_path = Path(source_path)
    output_index = _ensure_output_index(
        source_path,
        index=index,
        metadata_scope=metadata_scope,
    )

    source_ids = _collect_qualified_ids(output_index)
    if qualified_ids is not None:
        effective_ids = source_ids & qualified_ids
    else:
        effective_ids = source_ids

    selected_qualified_ids = set(
        _prepare_split_candidates(effective_ids, sample=sample)
    )
    id_splits = split_qualified_ids(
        effective_ids,
        ratios,
        seed=seed,
        sample=sample,
    )
    assigned_qualified_ids = set().union(*id_splits) if id_splits else set()
    unassigned_qualified_ids = selected_qualified_ids - assigned_qualified_ids

    split_payloads: list[dict[str, Any]] = []
    for split_name, ratio, split_ids in zip(normalized_names, ratios, id_splits):
        split_payloads.append(
            _build_split_index_payload(
                source_path,
                output_index,
                split_name,
                split_ids,
                ratio=ratio,
                seed=seed,
                sample=sample,
                metadata_scope=metadata_scope,
            )
        )
    split_results = _write_split_payloads_batch(
        source_path,
        split_payloads,
        metadata_scope=metadata_scope,
    )

    return {
        "source": str(source_path),
        "total_ids": len(source_ids),
        "selected_ids": len(selected_qualified_ids),
        "excluded_ids": len(source_ids - effective_ids),
        "selected_qualified_ids": selected_qualified_ids,
        "unassigned_qualified_ids": unassigned_qualified_ids,
        "qualified_id_splits": id_splits,
        "splits": split_results,
    }


def copy_dataset_splits(
    source_path: str | Path,
    target_path: str | Path,
    *,
    split_names: list[str] | None = None,
    override: bool = False,
    metadata_scope: str | None = None,
    source_metadata_scope: str | None = None,
    target_metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Replicate inline splits from a source dataset onto a target dataset.

    For each requested split on *source_path*, the qualified IDs stored
    in the split artifact are used to filter the target dataset's index
    and write a matching ``.ds_crawler/split_<name>.json`` on
    *target_path*.  The result: the target carries the exact same
    file-id partition as the source, so ``train``/``val``/``test`` splits
    stay aligned across datasets of the same family (e.g. an RGB dataset
    and its depth sibling).

    The target's own ``index.json`` is used for filtering.  If it does
    not yet exist, it is built and saved.  Every qualified ID from the
    source split must appear in the target index — any missing ID
    raises ``ValueError`` loudly, so incomplete splits are never
    silently produced.

    Args:
        source_path: Dataset (directory or ``.zip``) whose splits are
            read.
        target_path: Dataset (directory or ``.zip``) to write the
            copied splits to.
        split_names: Names of splits to copy.  When ``None`` (default),
            every split found on *source_path* is copied.
        override: When ``False`` (default), raises ``ValueError`` if
            the target already has a split with the same name.  When
            ``True``, the existing target split is overwritten.

    Returns:
        A summary dict with keys:

        - ``source``: ``str`` path of the source dataset.
        - ``target``: ``str`` path of the target dataset.
        - ``splits``: list of per-split result dicts, each with
          ``split``, ``filename``, ``metadata_file``, ``path``,
          ``num_ids``, and ``overridden`` (bool).

    Raises:
        FileNotFoundError: If *source_path* has no splits at all, or a
            requested split does not exist on the source.
        ValueError: If any qualified ID from a source split is missing
            on the target, or if a target split already exists and
            ``override=False``, or if *split_names* is an empty list.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    resolved_source_scope = (
        source_metadata_scope
        if source_metadata_scope is not None
        else metadata_scope
    )
    resolved_target_scope = (
        target_metadata_scope
        if target_metadata_scope is not None
        else metadata_scope
    )

    available = list_dataset_splits(
        source_path,
        metadata_scope=resolved_source_scope,
    )
    if not available:
        raise FileNotFoundError(
            f"No inline splits found on source dataset {source_path}"
        )

    if split_names is None:
        requested = list(available)
    else:
        if not split_names:
            raise ValueError("split_names must be non-empty when provided")
        requested = _normalize_split_names(split_names)
        available_set = set(available)
        missing_on_source = [name for name in requested if name not in available_set]
        if missing_on_source:
            raise FileNotFoundError(
                f"Source dataset {source_path} has no splits named "
                f"{sorted(missing_on_source)}. Available splits: {available}"
            )

    existing_on_target = set(
        list_dataset_splits(target_path, metadata_scope=resolved_target_scope)
    )
    if not override:
        conflicts = sorted(name for name in requested if name in existing_on_target)
        if conflicts:
            raise ValueError(
                f"Target dataset {target_path} already has splits {conflicts}. "
                "Pass override=True to replace them."
            )

    target_index = _ensure_output_index(
        target_path,
        metadata_scope=resolved_target_scope,
    )
    target_qualified_ids = _collect_qualified_ids(target_index)

    split_payloads: list[dict[str, Any]] = []
    for split_name in requested:
        split_filename = get_split_filename(split_name)
        split_artifact = read_metadata_json(
            source_path,
            split_filename,
            metadata_scope=resolved_source_scope,
        )
        if split_artifact is None:
            raise FileNotFoundError(
                f"Split {split_name!r} not found on source dataset "
                f"{source_path} (expected {split_filename}"
                + (
                    f" in metadata_scope={resolved_source_scope!r})"
                    if resolved_source_scope is not None
                    else ")"
                )
            )
        validate_split_artifact(
            split_artifact, context=f"split[{split_name!r}]"
        )

        source_ids = _collect_qualified_ids(split_artifact)
        missing_ids = source_ids - target_qualified_ids
        if missing_ids:
            preview = sorted(missing_ids)[:10]
            suffix = (
                f" (showing first 10 of {len(missing_ids)})"
                if len(missing_ids) > 10 else ""
            )
            raise ValueError(
                f"Cannot copy split {split_name!r} from {source_path} to "
                f"{target_path}: {len(missing_ids)} / {len(source_ids)} "
                f"qualified ID(s) from the source split have no match on "
                f"the target. Examples: {preview}{suffix}"
            )

        logger.info(
            "Copying split %r from %s to %s (%d qualified IDs)",
            split_name, source_path, target_path, len(source_ids),
        )

        payload = _build_split_index_payload(
            target_path,
            target_index,
            split_name=split_name,
            split_ids=source_ids,
            execution={
                "copied_from": {
                    "source": str(source_path),
                    "split": split_name,
                    **(
                        {"metadata_scope": resolved_source_scope}
                        if resolved_source_scope is not None
                        else {}
                    ),
                }
            },
            metadata_scope=resolved_target_scope,
        )
        payload["overridden"] = split_name in existing_on_target
        split_payloads.append(payload)
    split_results = _write_split_payloads_batch(
        target_path,
        split_payloads,
        metadata_scope=resolved_target_scope,
    )

    return {
        "source": str(source_path),
        "target": str(target_path),
        "splits": split_results,
    }


def create_aligned_dataset_splits(
    source_paths: list[str | Path],
    split_names: list[str],
    ratios: list[int | float],
    *,
    seed: int | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Create matching inline split metadata across multiple datasets.

    Only IDs present in every source are included, so each split name maps
    to the same qualified-ID partition across all datasets.
    """
    if len(split_names) != len(ratios):
        raise ValueError(
            f"split_names and ratios must have the same length, "
            f"got {len(split_names)} and {len(ratios)}"
        )
    if not source_paths:
        raise ValueError("source_paths must be non-empty")

    normalized_names = _normalize_split_names(split_names)
    sources = [Path(p) for p in source_paths]

    indices: list[dict[str, Any]] = []
    per_source_ids: list[set[tuple[str, ...]]] = []
    for src in sources:
        index = _ensure_output_index(src, metadata_scope=metadata_scope)
        indices.append(index)
        qids = _collect_qualified_ids(index)
        per_source_ids.append(qids)
        logger.info(
            "Loaded index for %s: %d qualified IDs", src, len(qids),
        )

    common_ids = per_source_ids[0]
    for qids in per_source_ids[1:]:
        common_ids = common_ids & qids
    logger.info(
        "Inline split intersection across %d sources: %d common qualified IDs",
        len(sources), len(common_ids),
    )

    selected_qualified_ids = set(
        _prepare_split_candidates(common_ids, sample=sample)
    )
    id_splits = split_qualified_ids(
        common_ids,
        ratios,
        seed=seed,
        sample=sample,
    )
    assigned_qualified_ids = set().union(*id_splits) if id_splits else set()
    unassigned_qualified_ids = selected_qualified_ids - assigned_qualified_ids

    per_source_results: list[dict[str, Any]] = []
    for src, index, qids in zip(sources, indices, per_source_ids):
        split_payloads: list[dict[str, Any]] = []
        for split_name, ratio, split_ids in zip(normalized_names, ratios, id_splits):
            split_payloads.append(
                _build_split_index_payload(
                    src,
                    index,
                    split_name,
                    split_ids,
                    ratio=ratio,
                    seed=seed,
                    sample=sample,
                    metadata_scope=metadata_scope,
                )
            )
        split_results = _write_split_payloads_batch(
            src,
            split_payloads,
            metadata_scope=metadata_scope,
        )

        per_source_results.append({
            "source": str(src),
            "total_ids": len(qids),
            "selected_ids": len(selected_qualified_ids),
            "excluded_ids": len(qids - common_ids),
            "splits": split_results,
        })

    return {
        "common_ids": common_ids,
        "selected_qualified_ids": selected_qualified_ids,
        "unassigned_qualified_ids": unassigned_qualified_ids,
        "qualified_id_splits": id_splits,
        "per_source": per_source_results,
    }


def create_hierarchy_dataset_splits(
    source_paths: str | Path | list[str | Path],
    hierarchy_rules: dict[str, Any] | list[dict[str, Any]],
    *,
    exclusive: bool | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Create inline split metadata from hierarchy-path selection rules.

    Rules are evaluated against hierarchy-qualified IDs, where each candidate
    has the form ``(*hierarchy_keys, file_id)``. Clause values must match the
    exact hierarchy child keys stored in the index tree. For named hierarchy
    captures this means values such as ``"weather:fog"`` when the crawler
    config uses ``separator=":"``.

    Args:
        source_paths: One dataset path, or multiple aligned dataset paths.
            Directory and ``.zip`` datasets are both supported.
        hierarchy_rules: Either ``{"exclusive": bool, "splits": [...]}``
            or a bare list of split rules. Each split rule has ``name`` and
            ``clauses``; each clause has ``levelIndex`` (or ``level_index``)
            and ``values``.
        exclusive: Optional override for the rule object's exclusivity flag.
            When true, rules are evaluated in order and a file matched by an
            earlier rule cannot appear in a later rule.
        sample: Optional stride applied within each matched rule. In exclusive
            mode, an earlier rule reserves all of its pre-sampling matches so
            sampled-out files are not reassigned to later rules.

    Returns:
        A summary dict with the common qualified IDs, per-rule ID sets, and
        per-source write results. The function validates all rules and selected
        files before writing, so an invalid or empty later rule does not leave
        partially-created split artifacts behind.
    """
    resolved_exclusive, rules = _normalize_hierarchy_split_rules(
        hierarchy_rules,
        exclusive=exclusive,
    )

    if isinstance(source_paths, (str, Path)):
        sources = [Path(source_paths)]
    else:
        sources = [Path(path) for path in source_paths]
    if not sources:
        raise ValueError("source_paths must be non-empty")

    indices: list[dict[str, Any]] = []
    per_source_ids: list[set[tuple[str, ...]]] = []
    for src in sources:
        index = _ensure_output_index(src, metadata_scope=metadata_scope)
        indices.append(index)
        qids = _collect_qualified_ids(index)
        per_source_ids.append(qids)
        logger.info(
            "Loaded index for %s: %d qualified IDs", src, len(qids),
        )

    common_ids = set(per_source_ids[0])
    for qids in per_source_ids[1:]:
        common_ids &= qids
    logger.info(
        "Hierarchy split intersection across %d sources: %d common qualified IDs",
        len(sources), len(common_ids),
    )

    id_splits, matched_counts = _select_hierarchy_rule_splits(
        common_ids,
        rules,
        exclusive=resolved_exclusive,
        sample=sample,
    )
    selected_qualified_ids = set().union(*id_splits) if id_splits else set()
    unassigned_qualified_ids = common_ids - selected_qualified_ids

    per_source_results: list[dict[str, Any]] = []
    for src, index, qids in zip(sources, indices, per_source_ids):
        split_payloads: list[dict[str, Any]] = []
        for rule, split_ids, matched_count in zip(rules, id_splits, matched_counts):
            payload = _build_split_index_payload(
                src,
                index,
                rule.name,
                split_ids,
                sample=sample,
                execution={
                    "allocation_mode": "hierarchy_rules",
                    "hierarchy_clauses": rule.clauses_for_execution(),
                    "exclusive": resolved_exclusive,
                },
                metadata_scope=metadata_scope,
            )
            payload["matched_ids"] = matched_count
            split_payloads.append(payload)
        split_results = _write_split_payloads_batch(
            src,
            split_payloads,
            metadata_scope=metadata_scope,
        )

        per_source_results.append({
            "source": str(src),
            "total_ids": len(qids),
            "selected_ids": len(selected_qualified_ids),
            "excluded_ids": len(qids - common_ids),
            "splits": split_results,
        })

    return {
        "common_ids": common_ids,
        "selected_qualified_ids": selected_qualified_ids,
        "unassigned_qualified_ids": unassigned_qualified_ids,
        "qualified_id_splits": id_splits,
        "per_source": per_source_results,
    }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _resolve_dataset_source(
    source: str | Path | dict[str, Any],
    *,
    split: str | None = None,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Resolve a dataset source to a hydrated dataset index dict.

    When *source* is already a dict it is returned as-is (assumed to be a
    loaded dataset index object).  When it is a path, ``index.json`` is
    checked first; if absent, ``ds-crawler.json`` is used to index on the
    fly.  Raises ``FileNotFoundError`` if neither file exists.
    """
    if isinstance(source, dict):
        return source
    # Lazy import to avoid circular dependency (parser → traversal is fine,
    # but operations → parser closes the loop).
    if split is not None:
        return load_dataset_split(source, split, metadata_scope=metadata_scope)

    from .parser import index_dataset_from_path
    return index_dataset_from_path(source, metadata_scope=metadata_scope)


def align_datasets(
    *args: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Align multiple dataset modalities by file ID.

    Each positional argument is a dict with keys:

    - ``modality`` (str): Label for this modality (e.g. ``"rgb"``,
      ``"depth"``).
    - ``source``: Either a filesystem path (``str`` or ``Path``) to a
      dataset root, or an already-loaded dataset index dict.
    - ``split`` (str, optional): Load ``source`` through a named inline
      split stored as ``.ds_crawler/split_<name>.json``.
    - ``metadata_scope`` (str, optional): Load metadata from
      ``.ds_crawler/<metadata_scope>/``.

    When *source* is a path, the function first looks for an existing
    ``index.json``.  If none is found it looks for a ``ds-crawler.json``
    configuration and indexes the dataset on the fly.  If neither file
    exists a ``FileNotFoundError`` is raised.

    Returns:
        A dict keyed by file ID.  Each value is a dict mapping modality
        labels to their corresponding file entry dicts.  IDs that are
        not present in every modality will have fewer keys than the
        number of input modalities.

    Example::

        aligned = align_datasets(
            {"modality": "rgb", "source": "/data/rgb"},
            {"modality": "depth", "source": depth_output_dict},
        )
        for file_id, modalities in aligned.items():
            if "rgb" in modalities and "depth" in modalities:
                rgb_path = modalities["rgb"]["path"]
                depth_path = modalities["depth"]["path"]
    """
    if not args:
        return {}

    per_modality: dict[str, dict[str, dict[str, Any]]] = {}
    for arg in args:
        modality = arg["modality"]
        source = arg["source"]
        output = _resolve_dataset_source(
            source,
            split=arg.get("split"),
            metadata_scope=arg.get("metadata_scope"),
        )
        entries = _collect_file_entries_by_id(_get_index_node(output))
        per_modality[modality] = entries
        logger.info(
            "align_datasets: modality '%s' has %d file entries",
            modality, len(entries),
        )

    # Union of all IDs
    all_ids: set[str] = set()
    for entries in per_modality.values():
        all_ids.update(entries.keys())

    # Build aligned dict
    aligned: dict[str, dict[str, dict[str, Any]]] = {}
    for file_id in sorted(all_ids):
        entry: dict[str, dict[str, Any]] = {}
        for modality, entries in per_modality.items():
            if file_id in entries:
                entry[modality] = entries[file_id]
        aligned[file_id] = entry

    # Log alignment stats
    n_modalities = len(per_modality)
    n_complete = sum(1 for v in aligned.values() if len(v) == n_modalities)
    logger.info(
        "align_datasets: %d unique IDs, %d with all %d modalities",
        len(aligned), n_complete, n_modalities,
    )

    return aligned


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


def copy_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    index: dict[str, Any] | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
    input_metadata_scope: str | None = None,
    output_metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Copy files referenced in a dataset index to a new location.

    Preserves the relative directory structure.  If *index* is not
    provided, ``index.json`` is loaded from *input_path*.  The index
    is written as ``index.json`` in *output_path* so the copied dataset
    is self-contained.

    Args:
        input_path: Root directory or ``.zip`` archive of the source
            dataset.
        output_path: Root directory or ``.zip`` archive for the
            destination.  Created if it does not exist.  When the path
            ends with ``.zip`` the copied files are written into a ZIP
            archive instead of to the filesystem.
        index: A dataset output dict (as returned by ``index_dataset``).
            When ``None``, ``index.json`` is read from *input_path*.
        sample: When set, keep only every *sample*-th indexed data file
            (deterministic subsampling on sorted paths).

    Returns:
        A summary dict with keys ``copied`` (int), ``missing`` (int),
        and ``missing_files`` (list of relative paths that were not
        found in the source).
    """
    import contextlib
    import zipfile

    from .zip_utils import _detect_root_prefix, _matches_zip_stem

    input_path = Path(input_path)
    output_path = Path(output_path)
    resolved_input_scope = (
        input_metadata_scope
        if input_metadata_scope is not None
        else metadata_scope
    )
    resolved_output_scope = (
        output_metadata_scope
        if output_metadata_scope is not None
        else metadata_scope
    )
    zip_input = is_zip_path(input_path)
    zip_output = output_path.suffix.lower() == ".zip"

    if index is None:
        from .parser import index_dataset_from_path

        index = index_dataset_from_path(
            input_path,
            metadata_scope=resolved_input_scope,
        )

    assert index is not None  # ensured by the branch above
    all_paths = _collect_all_referenced_paths(index)

    if sample is not None and sample > 1:
        file_paths = sorted(set(get_files(index)))
        sampled_files = file_paths[::sample]
        all_paths = sampled_files
        index = _filter_index_by_paths(index, set(sampled_files))

    # Deduplicate while preserving order
    unique_paths = list(dict.fromkeys(all_paths))

    copied = 0
    missing = 0
    missing_files: list[str] = []

    # Prepare source zip context (if applicable)
    src_prefix = ""
    src_name_set: set[str] = set()

    with contextlib.ExitStack() as stack:
        src_zf: zipfile.ZipFile | None = None
        if zip_input:
            src_zf = stack.enter_context(zipfile.ZipFile(input_path, "r"))
            namelist = src_zf.namelist()
            src_prefix = _detect_root_prefix(namelist)
            if not _matches_zip_stem(src_prefix, input_path):
                src_prefix = ""
            src_name_set = set(namelist)

        dst_zf: zipfile.ZipFile | None = None
        if zip_output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dst_zf = stack.enter_context(
                zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED)
            )

        for rel_path_str in unique_paths:
            # --- read source ---
            if src_zf is not None:
                entry = (
                    src_prefix + rel_path_str if src_prefix else rel_path_str
                )
                entry = entry.replace("\\", "/")
                if entry not in src_name_set:
                    alt = rel_path_str.replace("\\", "/")
                    if alt in src_name_set:
                        entry = alt
                    else:
                        logger.warning(
                            "Source file not found in zip, skipping: %s",
                            rel_path_str,
                        )
                        missing += 1
                        missing_files.append(rel_path_str)
                        continue
                src_data = src_zf.read(entry)
            else:
                src = input_path / rel_path_str
                if not src.is_file():
                    logger.warning(
                        "Source file not found, skipping: %s", src
                    )
                    missing += 1
                    missing_files.append(rel_path_str)
                    continue
                src_data = None  # defer read; use shutil.copy2 when possible

            # --- write destination ---
            if dst_zf is not None:
                if src_data is None:
                    src_data = (input_path / rel_path_str).read_bytes()
                suffix = Path(rel_path_str).suffix.lower()
                compress = (
                    zipfile.ZIP_STORED
                    if suffix in _COMPRESSED_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                dst_zf.writestr(
                    rel_path_str.replace("\\", "/"),
                    src_data,
                    compress_type=compress,
                )
            else:
                dst = output_path / rel_path_str
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src_data is not None:
                    dst.write_bytes(src_data)
                else:
                    shutil.copy2(input_path / rel_path_str, dst)
            copied += 1

        head_file = str(index.get("head_file", DATASET_HEAD_FILENAME))
        config_payload = build_crawler_config_for_output(index)
        index_artifact = build_index_artifact(index)
        save_payload = {
            CONFIG_FILENAME: config_payload,
            OUTPUT_FILENAME: index_artifact,
        }
        if isinstance(index.get("head"), dict):
            save_payload[head_file] = index["head"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_output:
        output_path.mkdir(parents=True, exist_ok=True)
    write_metadata_json_batch(
        output_path,
        save_payload,
        metadata_scope=resolved_output_scope,
    )

    logger.info(
        "copy_dataset complete: %d files copied, %d missing", copied, missing
    )

    return {
        "copied": copied,
        "missing": missing,
        "missing_files": missing_files,
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def _load_required_index(
    dataset_path: Path,
    filename: str = OUTPUT_FILENAME,
    *,
    metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Load an output index from *dataset_path*, raising if absent."""
    if filename == OUTPUT_FILENAME:
        from .parser import index_dataset_from_path

        return index_dataset_from_path(
            dataset_path,
            metadata_scope=metadata_scope,
        )

    index = read_metadata_json(
        dataset_path,
        filename,
        metadata_scope=metadata_scope,
    )
    if index is None:
        raise FileNotFoundError(
            f"No {filename} found at {dataset_path}"
            + (
                f" in metadata_scope={metadata_scope!r}"
                if metadata_scope is not None
                else ""
            )
        )
    return index


def _derive_split_path(source: Path, suffix: str) -> Path:
    """Derive a target path by appending *suffix* to *source*.

    For directories: ``/data/kitti_rgb`` → ``/data/kitti_rgb_train``
    For ZIP files:   ``/data/kitti_rgb.zip`` → ``/data/kitti_rgb_train.zip``
    """
    if source.suffix.lower() == ".zip":
        return source.with_name(f"{source.stem}_{suffix}.zip")
    return source.with_name(f"{source.name}_{suffix}")


def split_dataset(
    source_path: str | Path,
    ratios: list[int | float],
    target_paths: list[str | Path],
    *,
    qualified_ids: set[tuple[str, ...]] | None = None,
    seed: int | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
    source_metadata_scope: str | None = None,
    target_metadata_scope: str | None = None,
) -> dict[str, Any]:
    """Split a dataset into multiple targets according to numeric ratios.

    Loads ``index.json`` from *source_path* (raises ``FileNotFoundError``
    if absent), partitions the file entries by their hierarchy-qualified
    IDs, and copies each partition to the corresponding target path — in
    the same way ``copy_dataset`` handles file transfer.

    An ``index.json`` containing only the partition's entries is written
    into each target.

    Args:
        source_path: Root directory or ``.zip`` archive of the source
            dataset.  Must contain an ``index.json``.
        ratios: Positive percentages (e.g. ``[80, 20]``) or fractions
            (e.g. ``[0.8, 0.2]``). Totals may be less than full coverage,
            leaving some selected IDs unassigned.
        target_paths: Destination directories (or ``.zip`` archives),
            one per ratio entry.
        qualified_ids: When provided, only these IDs are considered
            for splitting (the rest are ignored).  This is useful when
            splitting multiple aligned datasets by a common intersection
            of their IDs.
        seed: Random seed for shuffling IDs before splitting.  ``None``
            means deterministic sorted order without shuffling.
        sample: Optional stride applied before splitting. ``sample=5``
            keeps every fifth eligible ID from the sorted candidate pool.
        metadata_scope: Use the same scoped metadata namespace for reading
            the source index and writing target indices.
        source_metadata_scope: Optional source-only metadata namespace. Takes
            precedence over ``metadata_scope`` for reading.
        target_metadata_scope: Optional target-only metadata namespace. Takes
            precedence over ``metadata_scope`` for writing.

    Returns:
        A dict with keys:

        - ``splits``: list of per-target result dicts, each with keys
          ``target``, ``ratio``, ``num_ids``, ``copied``, ``missing``,
          ``missing_files``.
        - ``qualified_id_splits``: list of sets (one per target)
          containing the qualified IDs assigned to that split.  Useful
          for applying the same split to other aligned datasets.
        - ``selected_qualified_ids``: the eligible IDs after optional
          filtering and sampling, before ratio coverage is applied.
        - ``unassigned_qualified_ids``: selected IDs left out because the
          ratios sum to less than full coverage.

    Raises:
        FileNotFoundError: If ``index.json`` is missing in the source.
        ValueError: If *ratios* / *target_paths* are invalid.
    """
    if len(ratios) != len(target_paths):
        raise ValueError(
            f"ratios and target_paths must have the same length, "
            f"got {len(ratios)} and {len(target_paths)}"
        )

    source_path = Path(source_path)
    resolved_source_scope = (
        source_metadata_scope
        if source_metadata_scope is not None
        else metadata_scope
    )
    resolved_target_scope = (
        target_metadata_scope
        if target_metadata_scope is not None
        else metadata_scope
    )
    index = _load_required_index(
        source_path,
        metadata_scope=resolved_source_scope,
    )

    # Collect IDs from the source index
    source_ids = _collect_qualified_ids(index)
    if qualified_ids is not None:
        # Intersect with the provided set (keeps only IDs that exist
        # in *both* the source index and the caller's set)
        effective_ids = source_ids & qualified_ids
    else:
        effective_ids = source_ids

    # Split the IDs
    selected_qualified_ids = set(
        _prepare_split_candidates(effective_ids, sample=sample)
    )
    id_splits = split_qualified_ids(
        effective_ids,
        ratios,
        seed=seed,
        sample=sample,
    )
    assigned_qualified_ids = set().union(*id_splits) if id_splits else set()
    unassigned_qualified_ids = selected_qualified_ids - assigned_qualified_ids

    # Copy each split
    split_results: list[dict[str, Any]] = []
    for split_ids, target, ratio in zip(id_splits, target_paths, ratios):
        filtered_index = filter_index_by_qualified_ids(index, split_ids)
        result = copy_dataset(
            source_path,
            target,
            index=filtered_index,
            input_metadata_scope=resolved_source_scope,
            output_metadata_scope=resolved_target_scope,
        )
        result["target"] = str(target)
        result["ratio"] = ratio
        result["num_ids"] = len(split_ids)
        split_results.append(result)

    return {
        "splits": split_results,
        "selected_qualified_ids": selected_qualified_ids,
        "unassigned_qualified_ids": unassigned_qualified_ids,
        "qualified_id_splits": id_splits,
    }


def split_datasets(
    source_paths: list[str | Path],
    suffixes: list[str],
    ratios: list[int | float],
    *,
    seed: int | None = None,
    sample: int | None = None,
    metadata_scope: str | None = None,
    source_metadata_scopes: list[str | None] | None = None,
    target_metadata_scopes: list[str | None] | None = None,
) -> dict[str, Any]:
    """Split multiple aligned datasets using a common ID intersection.

    Loads ``index.json`` from each source, computes the intersection of
    their hierarchy-qualified IDs, partitions that intersection according
    to *ratios*, and copies each partition into a derived target path for
    every source dataset.

    Target paths are derived by appending each suffix to the source path:

    - Directory ``/data/kitti_rgb`` with suffix ``"train"``
      → ``/data/kitti_rgb_train``
    - ZIP ``/data/kitti_rgb.zip`` with suffix ``"train"``
      → ``/data/kitti_rgb_train.zip``

    When a source has IDs that are absent from other sources (i.e. not
    part of the intersection), they are logged but silently excluded so
    that every split contains only entries present in *all* modalities.

    Args:
        source_paths: Dataset root directories or ``.zip`` archives.
            Each must contain an ``index.json``.
        suffixes: One label per split (e.g. ``["train", "val"]``).
            Must have the same length as *ratios*.
        ratios: Positive percentages (e.g. ``[80, 20]``) or fractions
            (e.g. ``[0.8, 0.2]``). Totals may be less than full coverage,
            leaving some selected common IDs unassigned.
        seed: Random seed for shuffling IDs before splitting.  ``None``
            means deterministic sorted order without shuffling.
        sample: Optional stride applied before splitting. ``sample=5``
            keeps every fifth common ID from the sorted candidate pool.
        metadata_scope: Use the same scoped metadata namespace for every
            source and target.
        source_metadata_scopes: Optional per-source namespaces for reading
            each source index.
        target_metadata_scopes: Optional per-source namespaces for writing
            each source's derived target indices. When omitted, each target
            uses the matching source scope.

    Returns:
        A dict with keys:

        - ``common_ids``: the set of qualified IDs present in every
          source (the intersection).
        - ``selected_qualified_ids``: the common IDs after optional
          sampling, before ratio coverage is applied.
        - ``unassigned_qualified_ids``: selected common IDs left out
          because the ratios sum to less than full coverage.
        - ``qualified_id_splits``: list of sets (one per suffix)
          partitioning the common IDs.
        - ``per_source``: list of per-source result dicts (same order
          as *source_paths*), each containing ``source``,
          ``total_ids``, ``excluded_ids``, and ``splits`` (a list
          of per-suffix copy results with ``target``, ``suffix``,
          ``ratio``, ``num_ids``, ``copied``, ``missing``,
          ``missing_files``).

    Raises:
        FileNotFoundError: If ``index.json`` is missing in any source.
        ValueError: If lengths of *suffixes* and *ratios* differ, or
            if *source_paths* is empty.
    """
    if len(suffixes) != len(ratios):
        raise ValueError(
            f"suffixes and ratios must have the same length, "
            f"got {len(suffixes)} and {len(ratios)}"
        )
    if not source_paths:
        raise ValueError("source_paths must be non-empty")
    if (
        source_metadata_scopes is not None
        and len(source_metadata_scopes) != len(source_paths)
    ):
        raise ValueError(
            "source_metadata_scopes must have the same length as source_paths"
        )
    if (
        target_metadata_scopes is not None
        and len(target_metadata_scopes) != len(source_paths)
    ):
        raise ValueError(
            "target_metadata_scopes must have the same length as source_paths"
        )

    sources = [Path(p) for p in source_paths]
    resolved_source_scopes = [
        source_metadata_scopes[i]
        if source_metadata_scopes is not None
        else metadata_scope
        for i in range(len(sources))
    ]
    resolved_target_scopes = [
        target_metadata_scopes[i]
        if target_metadata_scopes is not None
        else resolved_source_scopes[i]
        for i in range(len(sources))
    ]

    # --- Load indices and collect qualified IDs per source ---
    indices: list[dict[str, Any]] = []
    per_source_ids: list[set[tuple[str, ...]]] = []
    for src, source_scope in zip(sources, resolved_source_scopes):
        index = _load_required_index(src, metadata_scope=source_scope)
        indices.append(index)
        qids = _collect_qualified_ids(index)
        per_source_ids.append(qids)
        logger.info(
            "Loaded index for %s: %d qualified IDs", src, len(qids),
        )

    # --- Compute intersection ---
    common_ids = per_source_ids[0]
    for qids in per_source_ids[1:]:
        common_ids = common_ids & qids
    logger.info(
        "Intersection across %d sources: %d common qualified IDs",
        len(sources), len(common_ids),
    )

    # --- Log per-source exclusions ---
    for src, qids in zip(sources, per_source_ids):
        excluded = qids - common_ids
        if excluded:
            logger.warning(
                "%s: %d / %d IDs not present in all sources (excluded "
                "from split)",
                src, len(excluded), len(qids),
            )

    # --- Split the common IDs ---
    selected_qualified_ids = set(
        _prepare_split_candidates(common_ids, sample=sample)
    )
    id_splits = split_qualified_ids(
        common_ids,
        ratios,
        seed=seed,
        sample=sample,
    )
    assigned_qualified_ids = set().union(*id_splits) if id_splits else set()
    unassigned_qualified_ids = selected_qualified_ids - assigned_qualified_ids

    # --- Copy each split for each source ---
    per_source_results: list[dict[str, Any]] = []
    for src, index, qids, source_scope, target_scope in zip(
        sources,
        indices,
        per_source_ids,
        resolved_source_scopes,
        resolved_target_scopes,
    ):
        source_split_results: list[dict[str, Any]] = []
        for split_ids, suffix, ratio in zip(id_splits, suffixes, ratios):
            target = _derive_split_path(src, suffix)
            filtered_index = filter_index_by_qualified_ids(index, split_ids)
            result = copy_dataset(
                src,
                target,
                index=filtered_index,
                input_metadata_scope=source_scope,
                output_metadata_scope=target_scope,
            )
            result["target"] = str(target)
            result["suffix"] = suffix
            result["ratio"] = ratio
            result["num_ids"] = len(split_ids)
            source_split_results.append(result)

        per_source_results.append({
            "source": str(src),
            "total_ids": len(qids),
            "excluded_ids": len(qids - common_ids),
            "splits": source_split_results,
            "source_metadata_scope": source_scope,
            "target_metadata_scope": target_scope,
        })

    return {
        "common_ids": common_ids,
        "selected_qualified_ids": selected_qualified_ids,
        "unassigned_qualified_ids": unassigned_qualified_ids,
        "qualified_id_splits": id_splits,
        "per_source": per_source_results,
    }


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def extract_datasets(
    configs: list[dict[str, Any]],
    output_paths: list[str | Path],
    *,
    strict: bool = False,
    sample: int | None = None,
    match_index: dict[str, Any] | None = None,
    metadata_scope: str | None = None,
    metadata_scopes: list[str | None] | None = None,
) -> dict[str, Any]:
    """Extract multiple datasets from source directories using per-config regex patterns.

    Each config dict defines regex patterns that select specific files from
    its ``path`` directory.  The matched files are indexed and then copied
    to the corresponding output path.  An ``index.json`` is written in
    each target so the extracted dataset is self-contained.

    This is useful when a single source directory contains multiple
    modalities (e.g. both RGB and depth files) and each config selects
    one modality via different regex patterns.

    After extraction, the function computes the intersection of qualified
    IDs across all configs and logs a warning for any IDs that are present
    in some configs but missing from others.

    Args:
        configs: List of dataset configuration dicts (same shape as
            entries in ``config.json["datasets"]``).  Each must have a
            ``path`` key.
        output_paths: Destination directories (or ``.zip`` archives),
            one per config entry.  Created if they do not exist.
        strict: Abort on duplicate IDs or excessive regex misses during
            indexing.
        sample: Keep only every *sample*-th regex-matched file during
            indexing (deterministic subsampling).
        match_index: External filter -- only files whose ID appears in
            this index are included.
        metadata_scope: Use the same scoped metadata namespace for each
            config and output.
        metadata_scopes: Optional per-config namespaces. Takes precedence
            over ``metadata_scope`` and config-embedded ``metadata_scope``.

    Returns:
        A dict with keys:

        - ``extractions``: list of per-config result dicts, each with
          ``config_name``, ``source``, ``target``, ``num_ids``,
          ``copied``, ``missing``, ``missing_files``.
        - ``per_config_ids``: list of sets of qualified IDs per config.
        - ``common_ids``: the intersection of qualified IDs across all
          configs.
        - ``incomplete_ids``: dict mapping config name to the set of
          qualified IDs that are in that config but not in the
          intersection.  Empty when all configs match the same IDs.

    Raises:
        ValueError: If *configs* and *output_paths* have different
            lengths, or if *configs* is empty.
    """
    if len(configs) != len(output_paths):
        raise ValueError(
            f"configs and output_paths must have the same length, "
            f"got {len(configs)} and {len(output_paths)}"
        )
    if not configs:
        raise ValueError("configs must be non-empty")
    if metadata_scopes is not None and len(metadata_scopes) != len(configs):
        raise ValueError(
            "metadata_scopes must have the same length as configs"
        )

    from .parser import index_dataset

    resolved_scopes = [
        metadata_scopes[i]
        if metadata_scopes is not None
        else (
            metadata_scope
            if metadata_scope is not None
            else config.get("metadata_scope")
        )
        for i, config in enumerate(configs)
    ]

    # --- Index each config ---
    indices: list[dict[str, Any]] = []
    per_config_ids: list[set[tuple[str, ...]]] = []

    for i, (config, config_scope) in enumerate(zip(configs, resolved_scopes)):
        config_name = config.get("name", "unnamed")
        logger.info(
            "extract_datasets: indexing config %d/%d ('%s') from %s",
            i + 1, len(configs), config_name, config.get("path", "?"),
        )
        index = index_dataset(
            config,
            strict=strict,
            sample=sample,
            match_index=match_index,
            metadata_scope=config_scope,
        )
        indices.append(index)
        qids = _collect_qualified_ids(index)
        per_config_ids.append(qids)
        logger.info(
            "extract_datasets: config '%s' matched %d qualified IDs",
            config_name, len(qids),
        )

    # --- Compute intersection and warn about incomplete coverage ---
    common_ids = per_config_ids[0].copy()
    for qids in per_config_ids[1:]:
        common_ids = common_ids & qids
    logger.info(
        "extract_datasets: intersection across %d configs: %d common "
        "qualified IDs",
        len(configs), len(common_ids),
    )

    incomplete_ids: dict[str, set[tuple[str, ...]]] = {}
    for config, qids in zip(configs, per_config_ids):
        config_name = config.get("name", "unnamed")
        diff = qids - common_ids
        if diff:
            incomplete_ids[config_name] = diff
            logger.warning(
                "extract_datasets: config '%s' has %d IDs not present in "
                "all other configs (incomplete intersection)",
                config_name, len(diff),
            )

    # --- Copy each indexed dataset to its output path ---
    extraction_results: list[dict[str, Any]] = []

    for config, index, output_path, qids, config_scope in zip(
        configs, indices, output_paths, per_config_ids, resolved_scopes,
    ):
        config_name = config.get("name", "unnamed")
        source_path = config["path"]
        logger.info(
            "extract_datasets: copying '%s' from %s to %s",
            config_name, source_path, output_path,
        )
        result = copy_dataset(
            source_path,
            output_path,
            index=index,
            output_metadata_scope=config_scope,
        )
        result["config_name"] = config_name
        result["source"] = str(source_path)
        result["target"] = str(output_path)
        result["num_ids"] = len(qids)
        result["metadata_scope"] = config_scope
        extraction_results.append(result)

    return {
        "extractions": extraction_results,
        "per_config_ids": per_config_ids,
        "common_ids": common_ids,
        "incomplete_ids": incomplete_ids,
    }
