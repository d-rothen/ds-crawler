"""Sample a RealDriveSim dataset (RGB, depth, segmentation, calibration).

Indexes the RGB archive with deterministic subsampling, then copies a
matching subset from each modality into new ``*_sample_25.zip`` archives.

Usage::

    python sample_realdrivesim.py \
        --rgb_path       /path/to/rgb.zip \
        --depth_path     /path/to/depth.zip \
        --segmentation_path /path/to/segmentation.zip \
        --calibration_path  /path/to/calibration.zip
"""

import argparse
import json
import logging

from ds_crawler.parser import copy_dataset, index_dataset_from_path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _output_path(zip_path: str) -> str:
    """``/data/rgb.zip`` -> ``/data/rgb_sample_25.zip``."""
    assert zip_path.endswith(".zip"), f"Expected a .zip path, got: {zip_path}"
    return zip_path[: -len(".zip")] + "_sample_25.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsample a RealDriveSim dataset across modalities.",
    )
    parser.add_argument("--rgb_path", required=True, help="Path to the RGB zip archive.")
    parser.add_argument("--depth_path", required=True, help="Path to the depth zip archive.")
    parser.add_argument("--segmentation_path", required=True, help="Path to the segmentation zip archive.")
    parser.add_argument("--calibration_path", required=True, help="Path to the calibration zip archive.")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1) Index the RGB archive (every 25th file) and copy the subset.    #
    # ------------------------------------------------------------------ #
    log.info("Indexing RGB dataset: %s", args.rgb_path)
    rgb_index = index_dataset_from_path(
        args.rgb_path,
        strict=False,
        sample=25,
        save_index=False,
    )
    log.info("RGB index contains %d files.", len(json.dumps(rgb_index)))

    rgb_out = _output_path(args.rgb_path)
    log.info("Copying RGB subset -> %s", rgb_out)
    rgb_result = copy_dataset(args.rgb_path, rgb_out, index=rgb_index)
    log.info("RGB copy done: %s", rgb_result)

    # ------------------------------------------------------------------ #
    # 2) Index depth archive matched against the RGB index, then copy.   #
    # ------------------------------------------------------------------ #
    log.info("Indexing depth dataset: %s", args.depth_path)
    depth_index = index_dataset_from_path(
        args.depth_path,
        strict=False,
        save_index=False,
        match_index=rgb_index,
    )

    depth_out = _output_path(args.depth_path)
    log.info("Copying depth subset -> %s", depth_out)
    depth_result = copy_dataset(args.depth_path, depth_out, index=depth_index)
    log.info("Depth copy done: %s", depth_result)

    # ------------------------------------------------------------------ #
    # 3) Index segmentation archive matched against the RGB index, copy. #
    # ------------------------------------------------------------------ #
    log.info("Indexing segmentation dataset: %s", args.segmentation_path)
    seg_index = index_dataset_from_path(
        args.segmentation_path,
        strict=False,
        save_index=False,
        match_index=rgb_index,
    )

    seg_out = _output_path(args.segmentation_path)
    log.info("Copying segmentation subset -> %s", seg_out)
    seg_result = copy_dataset(args.segmentation_path, seg_out, index=seg_index)
    log.info("Segmentation copy done: %s", seg_result)

    # ------------------------------------------------------------------ #
    # 4) Read calibration archive (placeholder for future use).          #
    # ------------------------------------------------------------------ #
    log.info("Reading calibration dataset: %s", args.calibration_path)
    _cal_index = index_dataset_from_path(
        args.calibration_path,
        strict=False,
        save_index=False,
    )
    log.info("Calibration index loaded (%d chars). No further action.", len(json.dumps(_cal_index)))

    log.info("All done.")


if __name__ == "__main__":
    main()
