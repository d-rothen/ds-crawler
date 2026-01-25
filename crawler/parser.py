"""Dataset parsing logic."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import Config, DatasetConfig
from .handlers import get_handler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)
ANSI_DUPLICATE = "\033[31m"
ANSI_RESET = "\033[0m"
ID_MISS_PROMPT_RATIO = 0.2

ID_REGEX_JOIN_CHAR = "+"


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

    def __init__(self, config: Config) -> None:
        """Initialize parser with configuration."""
        self.config = config

    def parse_all(self) -> list[dict[str, Any]]:
        """Parse all configured datasets and return list of dataset outputs."""
        results = []

        for ds_config in self.config.datasets:
            dataset_node = self.parse_dataset(ds_config)
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

        # Deep merge dataset properties into the computed dataset_node
        if dataset_properties and isinstance(dataset_properties, dict):
            dataset_node = _deep_merge(dataset_node, dataset_properties)

        output = {
            "name": ds_config.name,
            "id_regex": ds_config.id_regex,
            "id_regex_join_char": ID_REGEX_JOIN_CHAR,
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

    def parse_dataset(self, ds_config: DatasetConfig) -> dict[str, Any]:
        """Parse a single dataset and return hierarchical DatasetNode."""
        handler_class = get_handler(ds_config.name)
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
        id_misses = 0
        prompted_for_id_miss = False
        id_miss_threshold = self._id_miss_threshold(len(files))
        matched_entries = 0

        # Process intrinsics files
        if ds_config.compiled_intrinsics_regex:
            logger.debug("Scanning for intrinsics files...")
        intrinsics_count = self._process_camera_files(
            files, base_path, ds_config, dataset_root, "intrinsics"
        )
        if intrinsics_count:
            logger.info(f"Processed {intrinsics_count} intrinsics files")

        # Process extrinsics files
        if ds_config.compiled_extrinsics_regex:
            logger.debug("Scanning for extrinsics files...")
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

                # Get hierarchy keys first (needed for per-level duplicate check)
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
                prompted_for_id_miss = self._prompt_on_id_miss(
                    id_misses,
                    len(files),
                    id_miss_threshold,
                    prompted_for_id_miss,
                )
                logger.debug(f"Skipped (id regex): {file_path}")
            elif skip_reason == "no_id":
                skipped_no_id += 1
                id_misses += 1
                prompted_for_id_miss = self._prompt_on_id_miss(
                    id_misses,
                    len(files),
                    id_miss_threshold,
                    prompted_for_id_miss,
                )
                logger.debug(f"Skipped (capture group was empty): {file_path}")
            elif skip_reason == "path_regex":
                skipped_path_regex += 1
                logger.debug(f"Skipped (path regex): {file_path}")

        logger.info(f"Processing complete. Matched {matched_entries}/{len(files)} entries into hierarchy")
        if duplicate_occurrences:
            logger.warning(
                "%sFound %d duplicate ids (%d extra entries)%s",
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

        return dataset_root

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
        """Process intrinsics or extrinsics files and place in hierarchy.

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

            # Read the camera data file
            try:
                camera_data = self._read_camera_file(file_path)
            except Exception as e:
                logger.error(f"Failed to read {camera_type} file {file_path}: {e}")
                continue

            # Place at the appropriate hierarchy level
            target_node = _ensure_hierarchy_path(dataset_root, hierarchy_keys)
            target_node[key_name] = camera_data
            count += 1

        return count

    def _read_camera_file(self, file_path: Path) -> Any:
        """Read camera intrinsics/extrinsics file.

        Supports JSON files. Can be extended for other formats.
        """
        with open(file_path, "r") as f:
            return json.load(f)

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
        basename_match = ds_config.compiled_basename_regex.match(basename)
        if not basename_match:
            return None, "basename"

        entry_properties = basename_match.groupdict()

        path_str = str(relative_path)

        # Extract id from relative path (search anywhere in the string)
        id_match = ds_config.compiled_id_regex.search(path_str)
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
            file_id = ID_REGEX_JOIN_CHAR.join(id_parts)
        else:
            groups = id_match.groups()
            if not groups or any(value is None for value in groups):
                return None, "no_id"
            file_id = ID_REGEX_JOIN_CHAR.join(groups)

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

    def _id_miss_threshold(self, total_files: int) -> int:
        return max(1, int(total_files * ID_MISS_PROMPT_RATIO))

    def _prompt_on_id_miss(
        self,
        id_misses: int,
        total_files: int,
        threshold: int,
        already_prompted: bool,
    ) -> bool:
        if already_prompted or total_files == 0:
            return already_prompted
        if id_misses <= threshold:
            return already_prompted
        if not self._confirm_continue(id_misses, total_files):
            raise RuntimeError("Aborted due to repeated id parse failures.")
        return True

    def _confirm_continue(self, id_misses: int, total_files: int) -> bool:
        try:
            response = input(
                f"Failed to parse {id_misses} out of {total_files} ids. Continue? (Y/N) "
            )
        except EOFError:
            return False
        return response.strip().lower() in {"y", "yes"}

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

        Returns list of output file paths that were written.
        """
        output_paths = []

        for ds_config in self.config.datasets:
            dataset_node = self.parse_dataset(ds_config)
            output = self._build_output(ds_config, dataset_node)

            output_path = Path(ds_config.path) / filename
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            output_paths.append(output_path)

        return output_paths
