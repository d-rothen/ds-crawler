#!/usr/bin/env python3
"""Dataset crawler main entry point."""

import argparse
import logging
import sys
from pathlib import Path

from ds_crawler.config import Config
from ds_crawler.parser import DatasetParser


def setup_logging(verbose: bool) -> None:
    """Configure logging with immediate flushing for cluster compatibility."""

    class FlushingHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    level = logging.DEBUG if verbose else logging.INFO
    handler = FlushingHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


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
    parser.add_argument(
        "-w",
        "--workdir",
        type=Path,
        default=None,
        help="Working directory to prepend to dataset paths.",
    )
    parser.add_argument(
        "-s",
        "--strict",
        action="store_true",
        help="Strict mode: abort on duplicate IDs or excessive regex misses.",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        config = Config.from_file(args.config, workdir=args.workdir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    parser_instance = DatasetParser(config, strict=args.strict)

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
