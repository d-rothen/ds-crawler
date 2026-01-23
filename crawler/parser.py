"""Dataset parsing logic."""

import json
import logging
from pathlib import Path
from typing import Any

from .config import Config, DatasetConfig
from .handlers import get_handler

logger = logging.getLogger(__name__)


class DatasetParser:
    """Parses datasets according to configuration."""

    def __init__(self, config: Config) -> None:
        """Initialize parser with configuration."""
        self.config = config

    def parse_all(self) -> dict[str, Any]:
        """Parse all configured datasets and return combined output."""
        all_entries = []

        for ds_config in self.config.datasets:
            entries = self.parse_dataset(ds_config)
            all_entries.extend(entries)

        return {"dataset": all_entries}

    def parse_dataset(self, ds_config: DatasetConfig) -> list[dict[str, Any]]:
        """Parse a single dataset and return its entries."""
        handler_class = get_handler(ds_config.name)
        handler = handler_class(ds_config)

        base_path = Path(ds_config.path)
        logger.info(f"Parsing dataset '{ds_config.name}' at: {base_path}")

        files = list(handler.get_files())
        logger.info(f"Found {len(files)} files")

        entries = []
        skipped_basename = 0
        skipped_no_id = 0
        skipped_path_regex = 0

        for file_path in files:
            entry, skip_reason = self._process_file(file_path, base_path, ds_config)
            if entry:
                entries.append(entry)
            elif skip_reason == "basename":
                skipped_basename += 1
                logger.debug(f"Skipped (basename regex): {file_path}")
            elif skip_reason == "no_id":
                skipped_no_id += 1
                logger.debug(f"Skipped (no id captured): {file_path}")
            elif skip_reason == "path_regex":
                skipped_path_regex += 1
                logger.debug(f"Skipped (path regex): {file_path}")

        logger.info(f"Matched {len(entries)} entries")
        if skipped_basename:
            logger.warning(f"Skipped {skipped_basename} files: basename regex did not match")
        if skipped_no_id:
            logger.warning(f"Skipped {skipped_no_id} files: 'id' capture group was empty")
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
            skip_reason indicates why: "basename", "no_id", or "path_regex".
        """
        relative_path = file_path.relative_to(base_path)
        basename = file_path.name

        # Extract entry properties from basename
        basename_match = ds_config.compiled_basename_regex.match(basename)
        if not basename_match:
            return None, "basename"

        entry_properties = basename_match.groupdict()

        # Validate 'id' was captured
        if "id" not in entry_properties or entry_properties["id"] is None:
            return None, "no_id"

        file_id = entry_properties["id"]

        # Extract path properties if path_regex is defined
        path_properties = {}
        if ds_config.compiled_path_regex:
            path_str = str(relative_path)
            path_match = ds_config.compiled_path_regex.match(path_str)
            if not path_match:
                return None, "path_regex"
            path_properties = path_match.groupdict()

        return {
            "name": ds_config.name,
            "type": ds_config.type,
            "gt": ds_config.gt,
            "path": str(relative_path),
            "id": file_id,
            "path_properties": path_properties,
            "entry_properties": entry_properties,
        }, None

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
            output = {"dataset": entries}

            output_path = Path(ds_config.path) / filename
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            output_paths.append(output_path)

        return output_paths
