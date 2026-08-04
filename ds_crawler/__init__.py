"""Dataset crawler package."""

from .artifact_builder import (
    build_crawler_config,
    build_dataset_artifacts_from_files,
    build_dataset_head,
)
from .config import Config, DatasetConfig, load_dataset_config
from .layout import (
    EULER_LAYOUT_ADDON,
    EULER_LAYOUT_VERSION,
    build_layout_addon,
    get_layout_addon,
    validate_layout_addon,
)
from .migration import (
    migrate_dataset_metadata,
    migrate_dataset_zip,
    migrate_dataset_zips_in_folder,
)
from .operations import (
    HierarchySplitClause,
    HierarchySplitRule,
    align_datasets,
    copy_dataset,
    copy_dataset_splits,
    create_aligned_dataset_splits,
    create_dataset_splits,
    create_hierarchy_dataset_splits,
    create_mapped_dataset_splits,
    extract_datasets,
    list_dataset_splits,
    load_dataset_split,
    split_dataset,
    split_datasets,
)
from .parser import (
    DatasetParser,
    index_dataset,
    index_dataset_from_files,
    index_dataset_from_path,
)
from .schema import (
    DatasetDescriptor,
    extract_dataset_properties,
    get_dataset_contract,
    get_dataset_properties,
)
from .traversal import (
    collect_qualified_ids,
    filter_index_by_qualified_ids,
    get_files,
    split_qualified_ids,
)
from .validation import validate_crawler_config, validate_dataset, validate_output
from .writer import DatasetWriter, ZipDatasetWriter
from .zip_utils import list_metadata_scopes, validate_metadata_scope

__all__ = [
    "DatasetDescriptor",
    "DatasetWriter",
    "ZipDatasetWriter",
    "Config",
    "DatasetConfig",
    "DatasetParser",
    "EULER_LAYOUT_ADDON",
    "EULER_LAYOUT_VERSION",
    "align_datasets",
    "build_crawler_config",
    "build_dataset_artifacts_from_files",
    "build_dataset_head",
    "build_layout_addon",
    "collect_qualified_ids",
    "copy_dataset",
    "copy_dataset_splits",
    "create_aligned_dataset_splits",
    "create_dataset_splits",
    "create_hierarchy_dataset_splits",
    "create_mapped_dataset_splits",
    "extract_datasets",
    "HierarchySplitClause",
    "HierarchySplitRule",
    "filter_index_by_qualified_ids",
    "get_files",
    "get_dataset_contract",
    "get_dataset_properties",
    "get_layout_addon",
    "index_dataset",
    "index_dataset_from_files",
    "index_dataset_from_path",
    "list_dataset_splits",
    "list_metadata_scopes",
    "load_dataset_config",
    "load_dataset_split",
    "migrate_dataset_metadata",
    "migrate_dataset_zip",
    "migrate_dataset_zips_in_folder",
    "split_dataset",
    "split_datasets",
    "split_qualified_ids",
    "validate_crawler_config",
    "validate_dataset",
    "validate_metadata_scope",
    "validate_layout_addon",
    "validate_output",
    "extract_dataset_properties",
]
