"""Dataset parsing logic."""

import json
import logging
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

class DatasetParser:
    """Parses datasets according to configuration."""

    def __init__(self, config: Config) -> None:
        """Initialize parser with configuration."""
        self.config = config

    def parse_all(self) -> list[dict[str, Any]]:
        """Parse all configured datasets and return list of dataset outputs."""
        results = []

        for ds_config in self.config.datasets:
            entries = self.parse_dataset(ds_config)
            output = {
                "name": ds_config.name,
                "id_regex": ds_config.id_regex,
                "id_regex_join_char": ID_REGEX_JOIN_CHAR,
                **ds_config.properties,
                "dataset": entries,
            }
            results.append(output)

        return results

    def parse_dataset(self, ds_config: DatasetConfig) -> list[dict[str, Any]]:
        """Parse a single dataset and return its entries."""
        handler_class = get_handler(ds_config.name)
        handler = handler_class(ds_config)

        base_path = Path(ds_config.path)
        logger.info(
            "Parsing dataset '%s' (%s) at: %s",
            ds_config.name,
            ds_config.type,
            base_path,
        )

        files = list(handler.get_files())
        logger.info(f"Found {len(files)} files")

        entries = []
        seen_ids: set[str] = set()
        first_seen_paths: dict[str, str] = {}
        duplicate_ids: set[str] = set()
        duplicate_occurrences = 0
        skipped_basename = 0
        skipped_id_regex = 0
        skipped_no_id = 0
        skipped_path_regex = 0
        id_misses = 0
        prompted_for_id_miss = False
        id_miss_threshold = self._id_miss_threshold(len(files))

        progress_desc = f"{ds_config.name} files"
        for file_path in self._iter_with_progress(files, progress_desc):
            entry, skip_reason = self._process_file(file_path, base_path, ds_config)
            if entry:
                entry_id = entry["id"]
                if entry_id in seen_ids:
                    duplicate_occurrences += 1
                    duplicate_ids.add(entry_id)
                    first_path = first_seen_paths.get(entry_id, "unknown")
                    logger.warning(
                        "%sDuplicate id%s: %s (first: %s, again: %s)",
                        ANSI_DUPLICATE,
                        ANSI_RESET,
                        entry_id,
                        first_path,
                        entry["path"],
                    )
                else:
                    seen_ids.add(entry_id)
                    first_seen_paths[entry_id] = entry["path"]
                entries.append(entry)
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
            elif skip_reason == "path_regex"id_regex:
                skipped_path_regex += 1
                logger.debug(f"Skipped (path regex): {file_path}")

        logger.info(f"Matched {len(entries)} entries")
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

        return entries

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

        # Extract id from full relative path
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

        log_every = max(total // 10, 1)

        def generator():
            for index, item in enumerate(files, 1):
                if index == 1 or index == total or index % log_every == 0:
                    logger.info("%s: %d/%d", desc, index, total)
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
            entries = self.parse_dataset(ds_config)
            output = {
                "name": ds_config.name,
                "id_regex": ds_config.id_regex,
                "id_regex_join_char": ID_REGEX_JOIN_CHAR,
                **ds_config.properties,
                "dataset": entries,
            }

            output_path = Path(ds_config.path) / filename
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            output_paths.append(output_path)

        return output_paths
