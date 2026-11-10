"""Imports from the shared ``euler-dataset-contract`` package."""

from __future__ import annotations

from euler_dataset_contract import (  # type: ignore[attr-defined]
    DATASET_CONTRACT_VERSION,
    DATASET_HEAD_KIND,
    MODALITY_META_SCHEMAS,
    DatasetHeadContract,
    build_default_meta,
    build_meta_schema,
    get_registered_addon_validators,
    normalize_meta_dict,
    parse_dataset_head,
    register_addon_validator,
    validate_addon_version,
    validate_contract_kind,
    validate_contract_version,
    validate_dataset_head,
    validate_meta_dict,
    validate_slot,
    validate_string_list,
    validate_token,
)

__all__ = [
    "DATASET_CONTRACT_VERSION",
    "DATASET_HEAD_KIND",
    "MODALITY_META_SCHEMAS",
    "DatasetHeadContract",
    "build_default_meta",
    "build_meta_schema",
    "get_registered_addon_validators",
    "normalize_meta_dict",
    "parse_dataset_head",
    "register_addon_validator",
    "validate_addon_version",
    "validate_contract_kind",
    "validate_contract_version",
    "validate_dataset_head",
    "validate_meta_dict",
    "validate_slot",
    "validate_string_list",
    "validate_token",
]

try:  # euler-dataset-contract >= 0.14
    from euler_dataset_contract import register_euler_loading_addon_validator
except ImportError:  # pragma: no cover - older contract
    register_euler_loading_addon_validator = None

if register_euler_loading_addon_validator is not None:
    # Dataset heads name their euler-loading decoder in addons.euler_loading.
    # Check them by the rules euler-loading loads them with, whether or not it
    # is installed, so a head ds-crawler writes is one euler-loading accepts.
    register_euler_loading_addon_validator()
