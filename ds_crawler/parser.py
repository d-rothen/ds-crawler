"""Dataset parsing logic."""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .config import Config, DatasetConfig, load_dataset_config
from .handlers import get_handler
from .zip_utils import (
    METADATA_DIR,
    OUTPUT_FILENAME,
    is_zip_path,
    read_metadata_json,
    write_metadata_json,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)
ANSI_DUPLICATE = "\033[31m"
ANSI_RESET = "\033[0m"
ID_MISS_WARN_RATIO = 0.2

# File extensions whose contents are already compressed.  Writing these
# with ZIP_STORED instead of ZIP_DEFLATED avoids a costly recompression
# pass that yields virtually no size reduction.
_COMPRESSED_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".exr", ".webp",
})


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict.

    - Dicts are recursively merged
    - Non-dict values from override replace base values
    - Keys only in base or only in override are preserved
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_hierarchy_keys(
    match: re.Match,
    separator: str | None,
) -> list[str]:
    """Extract hierarchy keys from regex match.

    For named capture groups: returns ['{name}{separator}{value}', ...]
    For simple capture groups: returns ['{value}', ...]
    """
    group_index = match.re.groupindex
    keys = []

    if group_index:
        # Named capture groups - sorted by their position in the pattern
        ordered_names = sorted(group_index.items(), key=lambda item: item[1])
        for name, _ in ordered_names:
            value = match.group(name)
            if value is not None:
                keys.append(f"{name}{separator}{value}")
    else:
        # Simple capture groups
        for value in match.groups():
            if value is not None:
                keys.append(value)

    return keys


def _ensure_hierarchy_path(root: dict, keys: list[str]) -> dict:
    """Navigate/create hierarchy path and return the target node.

    Creates 'children' dicts as needed along the path.
    Returns the node at the end of the path.
    """
    current = root
    for key in keys:
        if "children" not in current:
            current["children"] = {}
        if key not in current["children"]:
            current["children"][key] = {}
        current = current["children"][key]
    return current


def _add_file_to_node(node: dict, entry: dict) -> None:
    """Add a file entry to a node's files list."""
    if "files" not in node:
        node["files"] = []
    node["files"].append(entry)


class DatasetParser:
    """Parses datasets according to configuration."""

    def __init__(
        self,
        config: Config,
        strict: bool = False,
        sample: int | None = None,
        match_index: dict[str, Any] | None = None,
    ) -> None:
        """Initialize parser with configuration.

        Args:
            config: Loaded configuration.
            strict: If True, abort on duplicate IDs or excessive regex
                misses instead of warning and continuing.
            sample: If set, keep only every *sample*-th regex-matched file
                (after sorting for deterministic ordering).
            match_index: If set, a dataset output dict whose file IDs are
                used as a filter — only files whose ID appears in the
                match index are included.
        """
        self.config = config
        self.strict = strict
        self.sample = sample
        self.match_index = match_index

    def parse_all(self) -> list[dict[str, Any]]:
        """Parse all configured datasets and return list of dataset outputs."""
        results = []

        for ds_config in self.config.datasets:
            dataset_node = self.parse_dataset(
                ds_config,
                sample=self.sample,
                match_index=self.match_index,
            )
            output = self._build_output(ds_config, dataset_node)
            results.append(output)

        return results

    def _build_output(
        self, ds_config: DatasetConfig, dataset_node: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the output dict for a dataset."""
        # Separate dataset-level properties from top-level properties
        properties = ds_config.properties.copy()
        dataset_properties = properties.pop("dataset", None)
        # euler_train is normalized and emitted explicitly below.
        properties.pop("euler_train", None)

        # Deep merge dataset properties into the computed dataset_node
        if dataset_properties and isinstance(dataset_properties, dict):
            dataset_node = _deep_merge(dataset_node, dataset_properties)

        output = {
            "name": ds_config.name,
            "type": ds_config.type,
            "id_regex": ds_config.id_regex,
            "id_regex_join_char": ds_config.id_regex_join_char,
            "euler_train": ds_config.euler_train,
            **properties,
            "dataset": dataset_node,
        }
        if ds_config.hierarchy_regex:
            output["hierarchy_regex"] = ds_config.hierarchy_regex
        if ds_config.named_capture_group_value_separator:
            output["named_capture_group_value_separator"] = (
                ds_config.named_capture_group_value_separator
            )
        return output

    def parse_dataset(
        self,
        ds_config: DatasetConfig,
        *,
        sample: int | None = None,
        match_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Parse a single dataset and return hierarchical DatasetNode.

        Args:
            ds_config: Dataset configuration.
            sample: Keep every *sample*-th matched file.  Overrides the
                instance default when provided.
            match_index: Base output dict used to filter by file ID.
                Overrides the instance default when provided.
        """
        handler_class = get_handler(ds_config.name, path=ds_config.path)
        handler = handler_class(ds_config)

        base_path = Path(ds_config.path)
        logger.info(
            "Parsing dataset '%s' (%s) at: %s",
            ds_config.name,
            ds_config.type,
            base_path,
        )

        logger.info("Scanning directory for files...")
        files = list(handler.get_files())
        logger.info(f"File scan complete. Found {len(files)} files")

        return self._parse_files(
            ds_config, files, base_path,
            sample=sample, match_index=match_index,
        )

    def parse_dataset_from_files(
        self,
        ds_config: DatasetConfig,
        files: Iterable[str | Path],
        base_path: str | Path | None = None,
        *,
        sample: int | None = None,
        match_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Parse pre-collected files without handler-based file discovery.

        This bypasses the handler's ``get_files()`` step and processes the
        supplied *files* directly.  Useful when the caller already has a
        list of file paths (e.g. from an external source or a virtual
        dataset).

        Args:
            ds_config: Dataset configuration (regex patterns, properties, …).
            files: File paths to process.  Can be a list, generator, or any
                iterable of ``str`` or ``Path`` objects.
            base_path: Root path used to compute relative paths for regex
                matching.  Defaults to ``ds_config.path`` when *None*.
            sample: Keep every *sample*-th matched file.  Overrides the
                instance default when provided.
            match_index: Base output dict used to filter by file ID.
                Overrides the instance default when provided.

        Returns:
            Hierarchical dataset node (same structure as ``parse_dataset``).
        """
        resolved_base = Path(base_path) if base_path is not None else Path(ds_config.path)
        file_list = [Path(f) for f in files]

        logger.info(
            "Parsing dataset '%s' (%s) from %d pre-collected files (base: %s)",
            ds_config.name,
            ds_config.type,
            len(file_list),
            resolved_base,
        )

        return self._parse_files(
            ds_config, file_list, resolved_base,
            sample=sample, match_index=match_index,
        )

    def _parse_files(
        self,
        ds_config: DatasetConfig,
        files: list[Path],
        base_path: Path,
        *,
        sample: int | None = None,
        match_index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Core file-processing loop shared by all parse entry-points."""
        # Sort files for deterministic ordering (important for sampling)
        files = sorted(files)

        # Pre-compute match IDs from match_index (hierarchy-qualified)
        match_ids: set[tuple[str, ...]] | None = None
        if match_index is not None:
            match_ids = _collect_qualified_ids(match_index)
            logger.info("match_index filter active: %d qualified IDs to match against", len(match_ids))

        if sample is not None:
            logger.info("Sampling active: keeping every %d-th matched file", sample)

        # Root of the hierarchical structure
        dataset_root: dict[str, Any] = {}

        # Duplicate tracking - either global (flat_ids_unique=True) or per-hierarchy-level
        seen_ids: set[str] = set()
        first_seen_paths: dict[str, str] = {}
        seen_ids_per_level: dict[tuple[str, ...], set[str]] = {}
        first_seen_paths_per_level: dict[tuple[str, ...], dict[str, str]] = {}
        duplicate_ids: set[str] = set()
        duplicate_occurrences = 0
        skipped_basename = 0
        skipped_id_regex = 0
        skipped_no_id = 0
        skipped_path_regex = 0
        skipped_hierarchy = 0
        skipped_match_index = 0
        skipped_sample = 0
        id_misses = 0
        warned_id_miss = False
        id_miss_threshold = max(1, int(len(files) * ID_MISS_WARN_RATIO))
        matched_entries = 0
        sample_counter = 0

        # Process intrinsics/extrinsics files (stores paths in hierarchy)
        intrinsics_count = self._process_camera_files(
            files, base_path, ds_config, dataset_root, "intrinsics"
        )
        if intrinsics_count:
            logger.info(f"Processed {intrinsics_count} intrinsics files")

        extrinsics_count = self._process_camera_files(
            files, base_path, ds_config, dataset_root, "extrinsics"
        )
        if extrinsics_count:
            logger.info(f"Processed {extrinsics_count} extrinsics files")

        logger.info("Processing files...")

        progress_desc = f"{ds_config.name} files"
        for file_path in self._iter_with_progress(files, progress_desc):
            entry, skip_reason = self._process_file(file_path, base_path, ds_config)
            if entry:
                entry_id = entry["id"]

                # Get hierarchy keys first (needed for qualified match
                # and per-level duplicate check)
                hierarchy_keys = self._get_entry_hierarchy_keys(
                    entry["path"], ds_config
                )
                if hierarchy_keys is None:
                    skipped_hierarchy += 1
                    logger.error(
                        "hierarchy_regex did not match for file that matched id_regex: %s",
                        entry["path"],
                    )
                    continue

                # match_index filter (hierarchy-qualified)
                if match_ids is not None:
                    qualified_id = tuple(hierarchy_keys) + (entry_id,)
                    if qualified_id not in match_ids:
                        skipped_match_index += 1
                        continue

                # Sampling filter (applied after match_index)
                if sample is not None:
                    sample_counter += 1
                    if (sample_counter - 1) % sample != 0:
                        skipped_sample += 1
                        continue

                # Duplicate check - either global or per-hierarchy-level
                is_duplicate = False
                first_path = "unknown"
                if ds_config.flat_ids_unique:
                    if entry_id in seen_ids:
                        is_duplicate = True
                        first_path = first_seen_paths.get(entry_id, "unknown")
                    else:
                        seen_ids.add(entry_id)
                        first_seen_paths[entry_id] = entry["path"]
                else:
                    level_key = tuple(hierarchy_keys)
                    if level_key not in seen_ids_per_level:
                        seen_ids_per_level[level_key] = set()
                        first_seen_paths_per_level[level_key] = {}

                    if entry_id in seen_ids_per_level[level_key]:
                        is_duplicate = True
                        first_path = first_seen_paths_per_level[level_key].get(
                            entry_id, "unknown"
                        )
                    else:
                        seen_ids_per_level[level_key].add(entry_id)
                        first_seen_paths_per_level[level_key][entry_id] = entry["path"]

                if is_duplicate:
                    if self.strict:
                        raise RuntimeError(
                            f"Duplicate id '{entry_id}' "
                            f"(first: {first_path}, again: {entry['path']})"
                        )
                    duplicate_occurrences += 1
                    duplicate_ids.add(entry_id)
                    logger.warning(
                        "%sDuplicate id%s: %s (first: %s, again: %s)",
                        ANSI_DUPLICATE,
                        ANSI_RESET,
                        entry_id,
                        first_path,
                        entry["path"],
                    )
                    continue  # Always skip duplicates from output

                # Place entry in hierarchy
                target_node = _ensure_hierarchy_path(dataset_root, hierarchy_keys)
                _add_file_to_node(target_node, entry)
                matched_entries += 1

            elif skip_reason == "basename":
                skipped_basename += 1
                logger.debug(f"Skipped (basename regex): {file_path}")
            elif skip_reason == "id_regex":
                skipped_id_regex += 1
                id_misses += 1
                warned_id_miss = self._check_id_miss_threshold(
                    id_misses, len(files), id_miss_threshold, warned_id_miss
                )
                logger.debug(f"Skipped (id regex): {file_path}")
            elif skip_reason == "no_id":
                skipped_no_id += 1
                id_misses += 1
                warned_id_miss = self._check_id_miss_threshold(
                    id_misses, len(files), id_miss_threshold, warned_id_miss
                )
                logger.debug(f"Skipped (capture group was empty): {file_path}")
            elif skip_reason == "path_regex":
                skipped_path_regex += 1
                logger.debug(f"Skipped (path regex): {file_path}")

        logger.info(f"Processing complete. Matched {matched_entries}/{len(files)} entries into hierarchy")
        if duplicate_occurrences:
            logger.warning(
                "%sFound %d duplicate ids (%d extra entries skipped)%s",
                ANSI_DUPLICATE,
                len(duplicate_ids),
                duplicate_occurrences,
                ANSI_RESET,
            )
        if skipped_basename:
            logger.warning(f"Skipped {skipped_basename} files: basename regex did not match")
        if skipped_id_regex:
            logger.warning(f"Skipped {skipped_id_regex} files: id regex did not match")
        if skipped_no_id:
            logger.warning(f"Skipped {skipped_no_id} files: capture group was empty")
        if skipped_path_regex:
            logger.warning(f"Skipped {skipped_path_regex} files: path regex did not match")
        if skipped_hierarchy:
            logger.error(
                f"Skipped {skipped_hierarchy} files: hierarchy_regex did not match "
                "(but id_regex did - check your regex configuration)"
            )
        if skipped_match_index:
            logger.info(f"Skipped {skipped_match_index} files: ID not in match_index")
        if skipped_sample:
            logger.info(f"Skipped {skipped_sample} files: sampling (every {sample}th)")

        return dataset_root

    def _check_id_miss_threshold(
        self,
        id_misses: int,
        total_files: int,
        threshold: int,
        already_warned: bool,
    ) -> bool:
        """Check if ID misses exceed threshold.

        In strict mode, raises RuntimeError. Otherwise logs a warning once.
        Returns updated warned state.
        """
        if already_warned or total_files == 0:
            return already_warned
        if id_misses <= threshold:
            return already_warned

        msg = (
            f"{id_misses}/{total_files} files failed ID extraction "
            f"(>{ID_MISS_WARN_RATIO * 100:.0f}% threshold). "
            "Check your id_regex configuration."
        )
        if self.strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return True

    def _get_entry_hierarchy_keys(
        self, path_str: str, ds_config: DatasetConfig
    ) -> list[str] | None:
        """Get hierarchy keys for a file entry.

        Returns None if hierarchy_regex doesn't match (which is an error condition
        when id_regex did match).
        """
        hierarchy_regex = ds_config.compiled_hierarchy_regex
        if not hierarchy_regex:
            # No hierarchy defined - use flat structure with empty keys
            return []

        match = hierarchy_regex.match(path_str)
        if not match:
            return None

        return _get_hierarchy_keys(match, ds_config.named_capture_group_value_separator)

    def _process_camera_files(
        self,
        files: list[Path],
        base_path: Path,
        ds_config: DatasetConfig,
        dataset_root: dict,
        camera_type: str,
    ) -> int:
        """Process intrinsics or extrinsics files and place paths in hierarchy.

        Matches files against the intrinsics/extrinsics regex, extracts
        hierarchy keys, and stores the relative file path at the appropriate
        hierarchy level. The file content is NOT read — the consumer is
        responsible for reading the file in whatever format it uses.

        Args:
            files: List of all files in the dataset
            base_path: Base path of the dataset
            ds_config: Dataset configuration
            dataset_root: Root node of the hierarchy
            camera_type: Either "intrinsics" or "extrinsics"

        Returns:
            Number of camera files processed
        """
        if camera_type == "intrinsics":
            regex = ds_config.compiled_intrinsics_regex
        elif camera_type == "extrinsics":
            regex = ds_config.compiled_extrinsics_regex
        else:
            return 0

        if not regex:
            return 0

        count = 0
        key_name = f"camera_{camera_type}"

        for file_path in files:
            relative_path = file_path.relative_to(base_path)
            path_str = str(relative_path)

            match = regex.match(path_str)
            if not match:
                continue

            # Get hierarchy keys from the match
            hierarchy_keys = _get_hierarchy_keys(
                match, ds_config.named_capture_group_value_separator
            )

            # Place the relative path at the appropriate hierarchy level
            target_node = _ensure_hierarchy_path(dataset_root, hierarchy_keys)
            target_node[key_name] = path_str
            count += 1

        return count

    def _process_file(
        self,
        file_path: Path,
        base_path: Path,
        ds_config: DatasetConfig,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Process a single file and extract properties.

        Returns:
            Tuple of (entry_dict, skip_reason). If entry_dict is None,
            skip_reason indicates why: "basename", "id_regex", "no_id", or "path_regex".
        """
        relative_path = file_path.relative_to(base_path)
        basename = file_path.name

        # Extract entry properties from basename
        entry_properties = {}
        if ds_config.compiled_basename_regex:
            basename_match = ds_config.compiled_basename_regex.match(basename)
            if not basename_match:
                return None, "basename"
            entry_properties = basename_match.groupdict()

        path_str = str(relative_path)

        # Extract id from relative path
        id_match = ds_config.compiled_id_regex.match(path_str)
        if not id_match:
            return None, "id_regex"

        group_index = id_match.re.groupindex
        if group_index:
            ordered_names = sorted(group_index.items(), key=lambda item: item[1])
            id_parts = []
            for name, _ in ordered_names:
                value = id_match.group(name)
                if value is None:
                    return None, "no_id"
                id_parts.append(f"{name}-{value}")
            if not id_parts:
                return None, "no_id"
            file_id = ds_config.id_regex_join_char.join(id_parts)
        else:
            groups = id_match.groups()
            if not groups or any(value is None for value in groups):
                return None, "no_id"
            file_id = ds_config.id_regex_join_char.join(groups)

        # Extract path properties if path_regex is defined
        path_properties = {}
        if ds_config.compiled_path_regex:
            path_match = ds_config.compiled_path_regex.match(path_str)
            if not path_match:
                return None, "path_regex"
            path_properties = path_match.groupdict()

        return {
            "path": str(relative_path),
            "id": file_id,
            "path_properties": path_properties,
            "basename_properties": entry_properties,
        }, None

    def _iter_with_progress(
        self,
        files: list[Path],
        desc: str,
    ):
        if tqdm:
            return tqdm(files, total=len(files), desc=desc, unit="file", leave=False)
        return self._iter_with_logging(files, desc)

    def _iter_with_logging(self, files: list[Path], desc: str):
        total = len(files)
        if total == 0:
            return iter(files)

        # Log more frequently in verbose/debug mode
        if logger.isEnabledFor(logging.DEBUG):
            log_every = max(total // 100, 100)  # Every 1% or every 100 files
        else:
            log_every = max(total // 10, 1)  # Every 10%

        def generator():
            for index, item in enumerate(files, 1):
                if index == 1 or index == total or index % log_every == 0:
                    logger.info("%s: %d/%d (%.1f%%)", desc, index, total, 100 * index / total)
                yield item

        return generator()

    def write_output(self, output_path: str | Path) -> None:
        """Parse all datasets and write output to a single JSON file."""
        output = self.parse_all()
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    def write_outputs_per_dataset(self, filename: str = "output.json") -> list[Path]:
        """Parse each dataset and write output to its root folder.

        When the dataset path is a ``.zip`` file the output is written
        *inside* the archive (unless ``output_json`` overrides the
        destination).

        Returns list of output file paths that were written.  For ZIP
        datasets the returned path is the ZIP file itself.
        """
        output_paths = []

        for ds_config in self.config.datasets:
            dataset_node = self.parse_dataset(ds_config)
            output = self._build_output(ds_config, dataset_node)

            ds_path = Path(ds_config.path)

            if ds_config.output_json:
                output_path = Path(ds_config.output_json)
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
            else:
                output_path = write_metadata_json(ds_path, filename, output)

            output_paths.append(output_path)

        return output_paths


def index_dataset(
    config: dict[str, Any],
    *,
    strict: bool = False,
    save_index: bool = False,
    sample: int | None = None,
    match_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index a single dataset and return its output dict.

    This is the main public API for programmatic use after installing the
    package.  The *config* dict has the same shape as one entry in
    ``config.json["datasets"]``.

    Args:
        config: A dataset configuration dict.
        strict: Abort on duplicate IDs or excessive regex misses.
        save_index: If True, persist the output as ``output.json`` in the
            dataset's root directory.
        sample: If set, keep only every *sample*-th regex-matched file
            (after sorting for deterministic ordering).
        match_index: If set, a dataset output dict whose file IDs are
            used as a filter — only files whose ID appears in the
            match index are included in the output.

    Returns:
        The output object (same structure as one element of ``output.json``).
    """
    ds_config = DatasetConfig.from_dict(config)
    parser = DatasetParser(Config(datasets=[ds_config]), strict=strict)
    dataset_node = parser.parse_dataset(
        ds_config, sample=sample, match_index=match_index,
    )
    output = parser._build_output(ds_config, dataset_node)
    if sample is not None:
        output["sampled"] = sample
    if save_index:
        _save_output(output, Path(ds_config.path))
    return output


def index_dataset_from_files(
    config: dict[str, Any],
    files: Iterable[str | Path],
    *,
    base_path: str | Path | None = None,
    strict: bool = False,
    sample: int | None = None,
    match_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index a dataset from pre-collected file paths.

    Bypasses handler-based file discovery and processes the supplied
    *files* directly.  This is useful when the caller already has a
    complete list of paths (e.g. from an external file listing, a
    database, or a virtual dataset that does not exist on disk).

    Args:
        config: A dataset configuration dict (same shape as one entry in
            ``config.json["datasets"]``).
        files: File paths to process.  Can be a list, generator, or any
            iterable of ``str`` or ``Path`` objects.
        base_path: Root path used to compute relative paths for regex
            matching.  Defaults to the ``path`` key in *config* when
            *None*.
        strict: Abort on duplicate IDs or excessive regex misses.
        sample: If set, keep only every *sample*-th regex-matched file
            (after sorting for deterministic ordering).
        match_index: If set, a dataset output dict whose file IDs are
            used as a filter — only files whose ID appears in the
            match index are included in the output.

    Returns:
        The output object (same structure as one element of ``output.json``).
    """
    ds_config = DatasetConfig.from_dict(config)
    parser = DatasetParser(Config(datasets=[ds_config]), strict=strict)
    dataset_node = parser.parse_dataset_from_files(
        ds_config, files, base_path=base_path,
        sample=sample, match_index=match_index,
    )
    output = parser._build_output(ds_config, dataset_node)
    if sample is not None:
        output["sampled"] = sample
    return output


def index_dataset_from_path(
    path: str | Path,
    *,
    strict: bool = False,
    save_index: bool = False,
    force_reindex: bool = False,
    sample: int | None = None,
    match_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index a dataset by path, loading config from ``ds-crawler.json``.

    Looks for a ``ds-crawler.json`` file inside *path* (or inside the
    ``.zip`` archive at *path*) and uses it as the dataset configuration.

    Args:
        path: Root directory (or ``.zip`` file) of the dataset.
        strict: Abort on duplicate IDs or excessive regex misses.
        save_index: If True, persist the output as ``output.json`` in the
            dataset's root directory (or inside the ZIP archive).
        force_reindex: If False (default) and ``output.json`` already exists
            in the dataset root (or inside the ZIP), read and return it
            without re-indexing.
        sample: If set, keep only every *sample*-th regex-matched file
            (after sorting for deterministic ordering).
        match_index: If set, a dataset output dict whose file IDs are
            used as a filter — only files whose ID appears in the
            match index are included in the output.

    Returns:
        The output object (same structure as one element of ``output.json``).
    """
    dataset_path = Path(path)

    if not force_reindex and sample is None and match_index is None:
        cached = _read_cached_output(dataset_path)
        if cached is not None:
            return cached

    ds_config = load_dataset_config({"path": str(path)})
    parser = DatasetParser(Config(datasets=[ds_config]), strict=strict)
    dataset_node = parser.parse_dataset(
        ds_config, sample=sample, match_index=match_index,
    )
    output = parser._build_output(ds_config, dataset_node)
    if sample is not None:
        output["sampled"] = sample
    if save_index:
        _save_output(output, Path(ds_config.path))
    return output


def _read_cached_output(
    dataset_path: Path, filename: str = "output.json"
) -> dict[str, Any] | None:
    """Return a previously written output dict, or ``None``."""
    cached = read_metadata_json(dataset_path, filename)
    if cached is not None:
        logger.info(
            "Found existing %s for %s, skipping reindex",
            filename,
            dataset_path,
        )
    return cached


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


def _save_output(
    output: dict[str, Any],
    dataset_path: Path,
    filename: str = "output.json",
) -> None:
    """Write an output dict to ``.ds_crawler/{filename}`` inside *dataset_path*.

    When *dataset_path* is a ``.zip`` file the entry is written inside
    the archive.
    """
    write_metadata_json(dataset_path, filename, output)


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


def split_qualified_ids(
    qualified_ids: set[tuple[str, ...]],
    ratios: list[int],
    *,
    seed: int | None = None,
) -> list[set[tuple[str, ...]]]:
    """Split qualified IDs into groups according to integer percentages.

    Args:
        qualified_ids: Set of ``(*hierarchy_keys, file_id)`` tuples.
        ratios: List of integers that must sum to 100.  Each entry
            specifies what percentage of IDs goes into that group.
        seed: Random seed for shuffling before splitting.  When ``None``
            the IDs are split in sorted order (deterministic but not
            randomised).

    Returns:
        A list of sets (one per ratio) partitioning *qualified_ids*.

    Raises:
        ValueError: If *ratios* do not sum to 100 or contain non-positive
            values.
    """
    if not ratios:
        raise ValueError("ratios must be non-empty")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"All ratios must be positive, got {ratios}")
    if sum(ratios) != 100:
        raise ValueError(
            f"Ratios must sum to 100, got {sum(ratios)} from {ratios}"
        )

    ordered = sorted(qualified_ids)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    total = len(ordered)
    # Compute cumulative boundaries to avoid rounding drift
    boundaries: list[int] = []
    cumsum = 0
    for ratio in ratios[:-1]:
        cumsum += ratio
        boundaries.append(round(total * cumsum / 100))
    boundaries.append(total)

    splits: list[set[tuple[str, ...]]] = []
    start = 0
    for end in boundaries:
        splits.append(set(ordered[start:end]))
        start = end

    return splits


def split_dataset(
    source_path: str | Path,
    ratios: list[int],
    target_paths: list[str | Path],
    *,
    qualified_ids: set[tuple[str, ...]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Split a dataset into multiple targets according to percentages.

    Loads ``output.json`` from *source_path* (raises ``FileNotFoundError``
    if absent), partitions the file entries by their hierarchy-qualified
    IDs, and copies each partition to the corresponding target path — in
    the same way ``copy_dataset`` handles file transfer.

    An ``output.json`` containing only the partition's entries is written
    into each target.

    Args:
        source_path: Root directory or ``.zip`` archive of the source
            dataset.  Must contain an ``output.json``.
        ratios: List of positive integers summing to 100.  Position *i*
            determines the percentage of IDs that go to
            ``target_paths[i]``.
        target_paths: Destination directories (or ``.zip`` archives),
            one per ratio entry.
        qualified_ids: When provided, only these IDs are considered
            for splitting (the rest are ignored).  This is useful when
            splitting multiple aligned datasets by a common intersection
            of their IDs.
        seed: Random seed for shuffling IDs before splitting.  ``None``
            means deterministic sorted order without shuffling.

    Returns:
        A dict with keys:

        - ``splits``: list of per-target result dicts, each with keys
          ``target``, ``ratio``, ``num_ids``, ``copied``, ``missing``,
          ``missing_files``.
        - ``qualified_id_splits``: list of sets (one per target)
          containing the qualified IDs assigned to that split.  Useful
          for applying the same split to other aligned datasets.

    Raises:
        FileNotFoundError: If ``output.json`` is missing in the source.
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
    id_splits = split_qualified_ids(effective_ids, ratios, seed=seed)

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
        "qualified_id_splits": id_splits,
    }


def _derive_split_path(source: Path, suffix: str) -> Path:
    """Derive a target path by appending *suffix* to *source*.

    For directories: ``/data/kitti_rgb`` → ``/data/kitti_rgb_train``
    For ZIP files:   ``/data/kitti_rgb.zip`` → ``/data/kitti_rgb_train.zip``
    """
    if source.suffix.lower() == ".zip":
        return source.with_name(f"{source.stem}_{suffix}.zip")
    return source.with_name(f"{source.name}_{suffix}")


def split_datasets(
    source_paths: list[str | Path],
    suffixes: list[str],
    ratios: list[int],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Split multiple aligned datasets using a common ID intersection.

    Loads ``output.json`` from each source, computes the intersection of
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
            Each must contain an ``output.json``.
        suffixes: One label per split (e.g. ``["train", "val"]``).
            Must have the same length as *ratios*.
        ratios: Positive integers summing to 100 (e.g. ``[80, 20]``).
            Must have the same length as *suffixes*.
        seed: Random seed for shuffling IDs before splitting.  ``None``
            means deterministic sorted order without shuffling.

    Returns:
        A dict with keys:

        - ``common_ids``: the set of qualified IDs present in every
          source (the intersection).
        - ``qualified_id_splits``: list of sets (one per suffix)
          partitioning the common IDs.
        - ``per_source``: list of per-source result dicts (same order
          as *source_paths*), each containing ``source``,
          ``total_ids``, ``excluded_ids``, and ``splits`` (a list
          of per-suffix copy results with ``target``, ``suffix``,
          ``ratio``, ``num_ids``, ``copied``, ``missing``,
          ``missing_files``).

    Raises:
        FileNotFoundError: If ``output.json`` is missing in any source.
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
    id_splits = split_qualified_ids(common_ids, ratios, seed=seed)

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
        "qualified_id_splits": id_splits,
        "per_source": per_source_results,
    }


def _load_required_index(
    dataset_path: Path, filename: str = "output.json"
) -> dict[str, Any]:
    """Load an output index from *dataset_path*, raising if absent."""
    index = read_metadata_json(dataset_path, filename)
    if index is None:
        raise FileNotFoundError(
            f"No {filename} found at {dataset_path}"
        )
    return index


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


def _resolve_dataset_source(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Resolve a dataset source to an output JSON dict.

    When *source* is already a dict it is returned as-is (assumed to be a
    loaded output JSON object).  When it is a path, ``output.json`` is
    checked first; if absent, ``ds-crawler.json`` is used to index on the
    fly.  Raises ``FileNotFoundError`` if neither file exists.
    """
    if isinstance(source, dict):
        return source
    return index_dataset_from_path(source)


def align_datasets(
    *args: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Align multiple dataset modalities by file ID.

    Each positional argument is a dict with keys:

    - ``modality`` (str): Label for this modality (e.g. ``"rgb"``,
      ``"depth"``).
    - ``source``: Either a filesystem path (``str`` or ``Path``) to a
      dataset root, or an already-loaded output JSON dict.

    When *source* is a path, the function first looks for an existing
    ``output.json``.  If none is found it looks for a ``ds-crawler.json``
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
        output = _resolve_dataset_source(source)
        entries = _collect_file_entries_by_id(output.get("dataset", {}))
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


def copy_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    index: dict[str, Any] | None = None,
    sample: int | None = None,
) -> dict[str, Any]:
    """Copy files referenced in a dataset index to a new location.

    Preserves the relative directory structure.  If *index* is not
    provided, ``output.json`` is loaded from *input_path*.  The index
    is written as ``output.json`` in *output_path* so the copied dataset
    is self-contained.

    Args:
        input_path: Root directory or ``.zip`` archive of the source
            dataset.
        output_path: Root directory or ``.zip`` archive for the
            destination.  Created if it does not exist.  When the path
            ends with ``.zip`` the copied files are written into a ZIP
            archive instead of to the filesystem.
        index: A dataset output dict (as returned by ``index_dataset``).
            When ``None``, ``output.json`` is read from *input_path*.
        sample: When set, keep only every *sample*-th data file
            (deterministic subsampling on sorted paths).  Camera files
            (intrinsics / extrinsics) are always copied.

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
        index = read_metadata_json(input_path, OUTPUT_FILENAME)
        if index is None:
            raise FileNotFoundError(
                f"No {OUTPUT_FILENAME} found at {input_path} and no index was provided"
            )

    assert index is not None  # ensured by the branch above
    all_paths = _collect_all_referenced_paths(index)

    if sample is not None and sample > 1:
        # Separate data-file paths from camera/auxiliary paths so that
        # sampling is applied only to data files.
        file_path_set = set(get_files(index))
        camera_paths = [p for p in all_paths if p not in file_path_set]
        file_paths = sorted(file_path_set)
        sampled_files = file_paths[::sample]
        all_paths = sampled_files + camera_paths
        # Filter the index to only contain the sampled file entries
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

        # Write the (possibly filtered) index
        if dst_zf is not None:
            dst_zf.writestr(
                f"{METADATA_DIR}/{OUTPUT_FILENAME}",
                json.dumps(index, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            write_metadata_json(output_path, OUTPUT_FILENAME, index)

    logger.info(
        "copy_dataset complete: %d files copied, %d missing", copied, missing
    )

    return {
        "copied": copied,
        "missing": missing,
        "missing_files": missing_files,
    }
