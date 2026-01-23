"""Dataset parsing logic."""

import json
import re
from pathlib import Path
from typing import Any

from .config import Config, DatasetConfig
from .handlers import get_handler


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
        entries = []

        for file_path in handler.get_files():
            entry = self._process_file(file_path, base_path, ds_config)
            if entry:
                entries.append(entry)

        return entries

    def _process_file(
        self,
        file_path: Path,
        base_path: Path,
        ds_config: DatasetConfig,
    ) -> dict[str, Any] | None:
        """Process a single file and extract properties."""
        relative_path = file_path.relative_to(base_path)
        basename = file_path.name

        # Extract entry properties from basename
        basename_match = ds_config.compiled_basename_regex.match(basename)
        if not basename_match:
            return None

        entry_properties = basename_match.groupdict()

        # Validate 'id' was captured
        if "id" not in entry_properties or entry_properties["id"] is None:
            return None

        file_id = entry_properties["id"]

        # Extract path properties if path_regex is defined
        path_properties = {}
        if ds_config.compiled_path_regex:
            path_str = str(relative_path)
            path_match = ds_config.compiled_path_regex.match(path_str)
            if path_match:
                path_properties = path_match.groupdict()

        return {
            "path": str(relative_path),
            "id": file_id,
            "path_properties": path_properties,
            "entry_properties": entry_properties,
        }

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
