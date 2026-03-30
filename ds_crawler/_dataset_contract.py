"""Import helpers for the sibling ``euler-dataset-contract`` package."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_package_on_path() -> None:
    candidate = Path(__file__).resolve().parents[2] / "euler-dataset-contract"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


try:
    from euler_dataset_contract import (  # type: ignore[attr-defined]
        DATASET_CONTRACT_VERSION,
        DATASET_HEAD_KIND,
        DatasetHeadContract,
        MODALITY_META_SCHEMAS,
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
except ModuleNotFoundError:
    _ensure_local_package_on_path()
    from euler_dataset_contract import (  # type: ignore[attr-defined]
        DATASET_CONTRACT_VERSION,
        DATASET_HEAD_KIND,
        DatasetHeadContract,
        MODALITY_META_SCHEMAS,
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
    "DatasetHeadContract",
    "MODALITY_META_SCHEMAS",
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
