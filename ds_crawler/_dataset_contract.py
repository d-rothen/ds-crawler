"""Import helpers for the sibling ``euler-dataset-contract`` package during extraction."""

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
        DatasetHeadContract,
        EULER_TRAIN_ALLOWED_KEYS,
        EULER_TRAIN_ALLOWED_USED_AS,
        KNOWN_NAMESPACE_KEYS,
        MODALITY_META_SCHEMAS,
        PROPERTY_NAMESPACE_KEYS,
        build_default_meta,
        build_meta_schema,
        fold_property_namespaces,
        is_namespace_key,
        normalize_euler_train,
        normalize_meta_dict,
        parse_dataset_head,
        validate_contract_version,
        validate_dataset_head,
        validate_euler_train,
        validate_meta_dict,
        validate_token,
    )
except ModuleNotFoundError:
    _ensure_local_package_on_path()
    from euler_dataset_contract import (  # type: ignore[attr-defined]
        DATASET_CONTRACT_VERSION,
        DatasetHeadContract,
        EULER_TRAIN_ALLOWED_KEYS,
        EULER_TRAIN_ALLOWED_USED_AS,
        KNOWN_NAMESPACE_KEYS,
        MODALITY_META_SCHEMAS,
        PROPERTY_NAMESPACE_KEYS,
        build_default_meta,
        build_meta_schema,
        fold_property_namespaces,
        is_namespace_key,
        normalize_euler_train,
        normalize_meta_dict,
        parse_dataset_head,
        validate_contract_version,
        validate_dataset_head,
        validate_euler_train,
        validate_meta_dict,
        validate_token,
    )


__all__ = [
    "DATASET_CONTRACT_VERSION",
    "DatasetHeadContract",
    "EULER_TRAIN_ALLOWED_KEYS",
    "EULER_TRAIN_ALLOWED_USED_AS",
    "KNOWN_NAMESPACE_KEYS",
    "MODALITY_META_SCHEMAS",
    "PROPERTY_NAMESPACE_KEYS",
    "build_default_meta",
    "build_meta_schema",
    "fold_property_namespaces",
    "is_namespace_key",
    "normalize_euler_train",
    "normalize_meta_dict",
    "parse_dataset_head",
    "validate_contract_version",
    "validate_dataset_head",
    "validate_euler_train",
    "validate_meta_dict",
    "validate_token",
]
