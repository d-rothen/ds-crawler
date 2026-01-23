#!/usr/bin/env python3
"""Dataset crawler main entry point."""

import argparse
import sys
from pathlib import Path

from crawler.config import Config
from crawler.parser import DatasetParser


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crawl datasets and extract metadata based on configuration."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Single output JSON file path. If not specified, writes output.json to each dataset's root folder.",
    )

    args = parser.parse_args()

    try:
        config = Config.from_file(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    parser_instance = DatasetParser(config)

    try:
        if args.output:
            parser_instance.write_output(args.output)
            print(f"Output written to: {args.output}")
        else:
            output_paths = parser_instance.write_outputs_per_dataset()
            for path in output_paths:
                print(f"Output written to: {path}")
    except Exception as e:
        print(f"Error processing datasets: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
