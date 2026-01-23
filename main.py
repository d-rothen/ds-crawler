#!/usr/bin/env python3
"""Dataset crawler main entry point."""

import argparse
import logging
import sys
from pathlib import Path

from crawler.config import Config
from crawler.parser import DatasetParser


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show each skipped file)",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        config = Config.from_file(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    parser_instance = DatasetParser(config)

    logger = logging.getLogger(__name__)

    try:
        if args.output:
            parser_instance.write_output(args.output)
            logger.info(f"Output written to: {args.output}")
        else:
            output_paths = parser_instance.write_outputs_per_dataset()
            for path in output_paths:
                logger.info(f"Output written to: {path}")
    except Exception as e:
        print(f"Error processing datasets: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
