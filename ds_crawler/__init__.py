"""Dataset crawler package."""

from .schema import DatasetDescriptor
from .config import Config, DatasetConfig, load_dataset_config
from .validation import validate_crawler_config, validate_dataset, validate_output
from .parser import (
    DatasetParser,
    index_dataset,
    index_dataset_from_files,
    index_dataset_from_path,
)
from .traversal import (
    collect_qualified_ids,
    filter_index_by_qualified_ids,
    get_files,
    split_qualified_ids,
)
from .operations import (
    align_datasets,
    copy_dataset,
    extract_datasets,
    split_dataset,
    split_datasets,
)
from .writer import DatasetWriter, ZipDatasetWriter

__all__ = [
    "DatasetDescriptor",
    "DatasetWriter",
    "ZipDatasetWriter",
    "Config",
    "DatasetConfig",
    "DatasetParser",
    "align_datasets",
    "collect_qualified_ids",
    "copy_dataset",
    "extract_datasets",
    "filter_index_by_qualified_ids",
    "get_files",
    "index_dataset",
    "index_dataset_from_files",
    "index_dataset_from_path",
    "load_dataset_config",
    "split_dataset",
    "split_datasets",
    "split_qualified_ids",
    "validate_crawler_config",
    "validate_dataset",
    "validate_output",
]
