#!/usr/bin/env python3
"""Dataset crawler main entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path

from ds_crawler.config import Config
from ds_crawler.migration import (
    migrate_dataset_metadata,
    migrate_dataset_zip,
    migrate_dataset_zips_in_folder,
)
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


def _run_index(argv: list[str]) -> int:
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
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Keep every Nth regex-matched file (deterministic subsampling).",
    )
    parser.add_argument(
        "--match-index",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to an output.json whose file IDs are used as a filter.",
    )

    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    try:
        config = Config.from_file(args.config, workdir=args.workdir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    match_index = None
    if args.match_index is not None:
        try:
            with open(args.match_index) as f:
                match_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading match-index: {e}", file=sys.stderr)
            return 1

    parser_instance = DatasetParser(
        config,
        strict=args.strict,
        sample=args.sample,
        match_index=match_index,
    )

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


def _run_migrate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy ds-crawler metadata to the new schema."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Dataset roots (directories or zip archives) to migrate",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not rewrite output.json; only write dataset-head.json and ds-crawler.json.",
    )
    parser.add_argument(
        "--scan-zips",
        action="store_true",
        help="When a path is a directory, scan it for .zip archives and attempt migration on each archive.",
    )
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="When used with --scan-zips, only scan the provided directory itself.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    args = parser.parse_args(argv)
    if args.top_level_only and not args.scan_zips:
        parser.error("--top-level-only requires --scan-zips")
    setup_logging(args.verbose)

    failed = False
    logger = logging.getLogger(__name__)
    for path in args.paths:
        try:
            if path.is_dir() and args.scan_zips:
                result = migrate_dataset_zips_in_folder(
                    path,
                    recursive=not args.top_level_only,
                    write_output=not args.no_output,
                    logger=logger,
                )
                if result["failed"]:
                    failed = True
                logger.info(
                    "Scanned %s (archives=%d, migrated=%d, failed=%d)",
                    result["path"],
                    result["scanned"],
                    len(result["migrated"]),
                    len(result["failed"]),
                )
                continue

            if path.suffix.lower() == ".zip":
                migrate_fn = migrate_dataset_zip
            else:
                migrate_fn = migrate_dataset_metadata
            result = migrate_fn(
                path,
                write_output=not args.no_output,
                logger=logger,
            )
        except Exception as exc:
            failed = True
            print(f"Migration failed for {path}: {exc}", file=sys.stderr)
            continue

        logger.info(
            "Migrated %s (head=%s, config=%s, output=%s, splits=%d)",
            result["path"],
            result["wrote_head"],
            result["wrote_config"],
            result["wrote_output"],
            len(result["migrated_splits"]),
        )

    return 1 if failed else 0


def main() -> int:
    """Main entry point."""
    argv = sys.argv[1:]
    if argv and argv[0] == "migrate-metadata":
        return _run_migrate(argv[1:])
    if argv and argv[0] == "index":
        return _run_index(argv[1:])
    return _run_index(argv)


if __name__ == "__main__":
    sys.exit(main())
