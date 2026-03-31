"""High-level dataset operations: align, copy, split.

These functions compose the traversal utilities from
:mod:`~ds_crawler.traversal` with the indexing API from
:mod:`~ds_crawler.parser` to provide bulk dataset manipulation.
"""

from __future__ import annotations

import json
import logging
import shutil
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


def _normalize_split_names(split_names: list[str]) -> list[str]:
    """Validate split names and reject duplicates."""
    normalized = [validate_split_name(name) for name in split_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("split_names must be unique")
    return normalized


def _ensure_output_index(
    dataset_path: Path,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a full output index, indexing and saving it when needed."""
    if index is not None:
        if read_metadata_json(dataset_path, OUTPUT_FILENAME) is None:
            save_output_artifacts(dataset_path, index)
        return index

    from .parser import index_dataset_from_path
    cached = read_metadata_json(dataset_path, OUTPUT_FILENAME)
    if cached is not None:
        return index_dataset_from_path(dataset_path)

    logger.info(
        "No %s found at %s, indexing dataset before writing splits",
        OUTPUT_FILENAME,
        dataset_path,
    )
    return index_dataset_from_path(dataset_path, save_index=True)


def _write_split_index(
    dataset_path: Path,
    index: dict[str, Any],
    split_name: str,
    split_ids: set[tuple[str, ...]],
    *,
    ratio: int | float | None = None,
    seed: int | None = None,
    sample: int | None = None,
) -> dict[str, Any]:
    """Persist a single split dataset artifact under ``.ds_crawler/``."""
    filename = get_split_filename(split_name)
    filtered_index = filter_index_by_qualified_ids(index, split_ids)
    split_execution: dict[str, Any] = {}
    if ratio is not None:
        split_execution["ratio"] = ratio
    if seed is not None:
        split_execution["seed"] = seed
    if sample is not None:
        split_execution["sampled"] = sample
    output_path = write_metadata_json(
        dataset_path,
        filename,
        build_split_artifact(
            filtered_index,
            split_name=split_name,
            execution=split_execution or None,
        ),
    )
    result: dict[str, Any] = {
        "split": split_name,
        "filename": filename,
        "metadata_file": f"{METADATA_DIR}/{filename}",
        "path": str(output_path),
        "num_ids": len(split_ids),
    }
    if ratio is not None:
        result["ratio"] = ratio
    return result


def list_dataset_splits(dataset_path: str | Path) -> list[str]:
    """Return sorted split names available for a dataset."""
    return list_split_names(Path(dataset_path))


def load_dataset_split(
    dataset_path: str | Path,
    split_name: str,
    *,
    strict: bool = False,
    save_index: bool = False,
    force_reindex: bool = False,
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
    )
    split_artifact = _load_required_index(dataset_path, get_split_filename(split_name))
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
    output_index = _ensure_output_index(source_path, index=index)

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

    split_results: list[dict[str, Any]] = []
    for split_name, ratio, split_ids in zip(normalized_names, ratios, id_splits):
        split_results.append(
            _write_split_index(
                source_path,
                output_index,
                split_name,
                split_ids,
                ratio=ratio,
                seed=seed,
                sample=sample,
            )
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


def create_aligned_dataset_splits(
    source_paths: list[str | Path],
    split_names: list[str],
    ratios: list[int | float],
    *,
    seed: int | None = None,
    sample: int | None = None,
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
        index = _ensure_output_index(src)
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
        split_results: list[dict[str, Any]] = []
        for split_name, ratio, split_ids in zip(normalized_names, ratios, id_splits):
            split_results.append(
                _write_split_index(
                    src,
                    index,
                    split_name,
                    split_ids,
                    ratio=ratio,
                    seed=seed,
                    sample=sample,
                )
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
        return load_dataset_split(source, split)

    from .parser import index_dataset_from_path
    return index_dataset_from_path(source)


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
        output = _resolve_dataset_source(source, split=arg.get("split"))
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
    zip_input = is_zip_path(input_path)
    zip_output = output_path.suffix.lower() == ".zip"

    if index is None:
        from .parser import index_dataset_from_path

        index = index_dataset_from_path(input_path)

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
        # Write the (possibly filtered) index
        if dst_zf is not None:
            if isinstance(index.get("head"), dict):
                dst_zf.writestr(
                    f"{METADATA_DIR}/{head_file}",
                    json.dumps(index["head"], indent=2),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            dst_zf.writestr(
                f"{METADATA_DIR}/{CONFIG_FILENAME}",
                json.dumps(config_payload, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            dst_zf.writestr(
                f"{METADATA_DIR}/{OUTPUT_FILENAME}",
                json.dumps(index_artifact, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            save_payload = {
                CONFIG_FILENAME: config_payload,
                OUTPUT_FILENAME: index_artifact,
            }
            if isinstance(index.get("head"), dict):
                save_payload[head_file] = index["head"]
            write_metadata_json_batch(output_path, save_payload)

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
    dataset_path: Path, filename: str = OUTPUT_FILENAME
) -> dict[str, Any]:
    """Load an output index from *dataset_path*, raising if absent."""
    if filename == OUTPUT_FILENAME:
        from .parser import index_dataset_from_path

        return index_dataset_from_path(dataset_path)

    index = read_metadata_json(dataset_path, filename)
    if index is None:
        raise FileNotFoundError(
            f"No {filename} found at {dataset_path}"
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
    index = _load_required_index(source_path)

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
        result = copy_dataset(source_path, target, index=filtered_index)
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

    sources = [Path(p) for p in source_paths]

    # --- Load indices and collect qualified IDs per source ---
    indices: list[dict[str, Any]] = []
    per_source_ids: list[set[tuple[str, ...]]] = []
    for src in sources:
        index = _load_required_index(src)
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
    for src, index, qids in zip(sources, indices, per_source_ids):
        source_split_results: list[dict[str, Any]] = []
        for split_ids, suffix, ratio in zip(id_splits, suffixes, ratios):
            target = _derive_split_path(src, suffix)
            filtered_index = filter_index_by_qualified_ids(index, split_ids)
            result = copy_dataset(src, target, index=filtered_index)
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

    from .parser import index_dataset

    # --- Index each config ---
    indices: list[dict[str, Any]] = []
    per_config_ids: list[set[tuple[str, ...]]] = []

    for i, config in enumerate(configs):
        config_name = config.get("name", "unnamed")
        logger.info(
            "extract_datasets: indexing config %d/%d ('%s') from %s",
            i + 1, len(configs), config_name, config.get("path", "?"),
        )
        index = index_dataset(
            config, strict=strict, sample=sample, match_index=match_index,
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

    for config, index, output_path, qids in zip(
        configs, indices, output_paths, per_config_ids,
    ):
        config_name = config.get("name", "unnamed")
        source_path = config["path"]
        logger.info(
            "extract_datasets: copying '%s' from %s to %s",
            config_name, source_path, output_path,
        )
        result = copy_dataset(source_path, output_path, index=index)
        result["config_name"] = config_name
        result["source"] = str(source_path)
        result["target"] = str(output_path)
        result["num_ids"] = len(qids)
        extraction_results.append(result)

    return {
        "extractions": extraction_results,
        "per_config_ids": per_config_ids,
        "common_ids": common_ids,
        "incomplete_ids": incomplete_ids,
    }
