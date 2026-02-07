"""Dataset parsing logic."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from .config import Config, DatasetConfig, load_dataset_config
from .handlers import get_handler
from .traversal import _collect_qualified_ids
from .zip_utils import (
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
