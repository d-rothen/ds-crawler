"""Dataset crawler package."""

from .schema import (
    DatasetDescriptor,
    extract_dataset_properties,
    get_dataset_contract,
    get_dataset_properties,
)
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
    create_aligned_dataset_splits,
    create_dataset_splits,
    copy_dataset,
    extract_datasets,
    list_dataset_splits,
    load_dataset_split,
    split_dataset,
    split_datasets,
)
from .migration import migrate_dataset_metadata
from .writer import DatasetWriter, ZipDatasetWriter

__all__ = [
    "DatasetDescriptor",
    "DatasetWriter",
    "ZipDatasetWriter",
    "Config",
    "DatasetConfig",
    "DatasetParser",
    "align_datasets",
    "create_aligned_dataset_splits",
    "create_dataset_splits",
    "collect_qualified_ids",
    "copy_dataset",
    "extract_datasets",
    "filter_index_by_qualified_ids",
    "get_files",
    "get_dataset_contract",
    "get_dataset_properties",
    "index_dataset",
    "index_dataset_from_files",
    "index_dataset_from_path",
    "list_dataset_splits",
    "load_dataset_config",
    "load_dataset_split",
    "migrate_dataset_metadata",
    "split_dataset",
    "split_datasets",
    "split_qualified_ids",
    "validate_crawler_config",
    "validate_dataset",
    "validate_output",
    "extract_dataset_properties",
]
