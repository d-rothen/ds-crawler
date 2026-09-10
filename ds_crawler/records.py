"""Validated metadata publication and content-addressed records, without executors.

Directory publications use immutable generations and one atomic pointer. ZIPs
are built privately and replaced as a unit. Both APIs require a single writer.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

from euler_dataset_contract import (
    MaterializedTransforms,
    OutputPlan,
    canonical_digest,
    canonical_json,
    parse_dataset_head,
    register_descriptor_validators,
    resolve_receipt,
    validate_artifact_receipt,
)
from euler_dataset_contract._descriptor_definitions import FULL_ID, check

from .artifacts import build_crawler_config_for_output, build_index_artifact
from .config import CONFIG_FILENAME
from .writer import DatasetWriter
from .zip_utils import (
    DATASET_HEAD_FILENAME,
    OUTPUT_FILENAME,
    get_metadata_entry_name,
    get_zip_root_prefix,
    read_metadata_json,
    write_metadata_json,
)


def bytes_digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def iter_entries(tree, prefix=()):
    for entry in tree.get("files", []):
        full_id = "/" + "/".join((*prefix, entry["id"]))
        check(full_id, FULL_ID, "index full_id")
        yield full_id, entry
    for key, child in tree.get("children", {}).items():
        yield from iter_entries(child, (*prefix, key))


def _safe_relative(path):
    if not isinstance(path, str) or "\\" in path or "\x00" in path:
        raise ValueError("record/artifact path must be relative POSIX")
    p = PurePosixPath(path)
    if (
        p.is_absolute()
        or not p.parts
        or any(x in {"", ".", ".."} for x in path.split("/"))
    ):
        raise ValueError("record/artifact path must stay within its scope")
    return path


def read_artifact(root, path):
    root = Path(root)
    path = _safe_relative(path)
    if root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as archive:
            return archive.read(get_zip_root_prefix(root) + path)
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact symlink escapes dataset root")
    return target.read_bytes()


def read_record(root, path, *, metadata_scope=None):
    _safe_relative(path)
    value = read_metadata_json(Path(root), path, metadata_scope=metadata_scope)
    if value is None:
        raise ValueError(f"missing receipt record: {path}")
    return value


def validate_output_records(
    root, output, *, records=None, metadata_scope=None, subset=False
):
    """Verify index/receipts/references and actual encoded artifact bytes.

    Crawler checks producer assertions and integrity; it does not certify tensor
    execution or source verification performed by the producer.
    """
    register_descriptor_validators()
    head = parse_dataset_head(output["head"])
    addon = head.get_addon("euler_transforms")
    if not addon or addon.get("version") != "2.0":
        for _, entry in iter_entries(output["index"]):
            read_artifact(root, entry["path"])
        return {}
    addon = MaterializedTransforms(addon)
    plan = OutputPlan(addon["plan"])
    if (
        head.dataset_id != plan["dataset_id"]
        or head.modality_key != plan["modality_key"]
    ):
        raise ValueError("materialized output identity disagrees with containing head")
    entries = list(iter_entries(output["index"]))
    if len({fid for fid, _ in entries}) != len(entries):
        raise ValueError("duplicate qualified output ID")
    if not subset and (
        addon["mapping"]["count"] != len(entries)
        or addon["mapping"]["index_digest"] != canonical_digest(output["index"])
    ):
        raise ValueError("materialized index count/digest mismatch")
    used = {}

    def get_record(path):
        value = (
            records[path]
            if records is not None and path in records
            else read_record(root, path, metadata_scope=metadata_scope)
        )
        used[path] = value
        return value

    for full_id, entry in entries:
        location = entry.get("attributes", {}).get("euler_transforms")
        if location is None:
            raise ValueError(f"missing receipt for {full_id}")
        receipt = resolve_receipt(location, get_record)
        validate_artifact_receipt(receipt, plan, full_id=full_id)
        if (
            bytes_digest(read_artifact(root, entry["path"]))
            != receipt["artifact"]["digest"]
        ):
            raise ValueError(f"encoded artifact digest mismatch for {full_id}")
    return used


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finalize_output(
    root, output, *, records=None, metadata_scope=None, expected_full_ids=None
):
    """Validate before publishing one coherent, immutable metadata generation."""
    root = Path(root)
    if root.suffix.lower() == ".zip":
        raise ValueError("use ValidatedDatasetWriter for atomic ZIP publication")
    output = deepcopy(output)
    ids = [fid for fid, _ in iter_entries(output["index"])]
    if expected_full_ids is not None and set(ids) != set(expected_full_ids):
        raise ValueError(
            "incomplete output: expected qualified IDs do not match successful writes"
        )
    records = validate_output_records(
        root, output, records=records, metadata_scope=metadata_scope
    )
    files = {
        DATASET_HEAD_FILENAME: output["head"],
        CONFIG_FILENAME: build_crawler_config_for_output(output),
        OUTPUT_FILENAME: build_index_artifact(output),
        **records,
    }
    digests = {path: canonical_digest(value) for path, value in files.items()}
    generation = canonical_digest(digests)[7:]
    metadata = root / get_metadata_entry_name(
        "publication.json", metadata_scope=metadata_scope
    )
    for path, value in files.items():
        _safe_relative(path)
        _atomic_bytes(
            metadata.parent / "generations" / generation / path,
            canonical_json(value).encode("utf-8"),
        )
    # Scope discovery is not a completion claim. Publish it before the pointer.
    if metadata_scope:
        write_metadata_json(
            root, "scope.json", {"version": "1.0"}, metadata_scope=metadata_scope
        )
    _atomic_bytes(
        metadata,
        canonical_json(
            {
                "version": "1.0",
                "generation": generation,
                "files": sorted(files),
                "digests": digests,
            }
        ).encode("utf-8"),
    )
    return metadata


class ValidatedDatasetWriter(DatasetWriter):
    """Commit bytes after successful encoding, then explicitly finalize metadata.

    A frozen OutputPlan enables strict materialization. Directory resume is
    idempotent by qualified output ID and complete receipt/artifact identity.
    ZIP has one owner and one finalization; closing an unfinished writer drops
    its private staging directory without publishing the archive.
    """

    def __init__(
        self,
        root,
        *,
        head,
        plan=None,
        expected_full_ids=None,
        metadata_scope=None,
        separator=":",
        resume=False,
        publication_validator=None,
    ):
        self._publication_validator = publication_validator
        self._destination = Path(root)
        self._zip_output = self._destination.suffix.lower() == ".zip"
        self._staging = None
        if self._zip_output:
            if resume or self._destination.exists():
                raise ValueError(
                    "ZIP writer requires a new destination and one finalization"
                )
            self._staging = tempfile.TemporaryDirectory(prefix="ds-crawler-")
            root = Path(self._staging.name)
        self._plan = OutputPlan(plan) if plan is not None else None
        self._expected_ids = (
            None if expected_full_ids is None else frozenset(expected_full_ids)
        )
        self._records = {}
        self._committed = {}
        self._finished = False
        super().__init__(
            root,
            head=deepcopy(head),
            metadata_scope=metadata_scope,
            separator=separator,
        )
        if not self._zip_output:
            from .artifacts import load_saved_output

            saved = read_metadata_json(
                self._root, "publication.json", metadata_scope=metadata_scope
            )
            if saved:
                if not resume:
                    raise ValueError("completed output exists; use resume=True")
                previous = load_saved_output(self._root, metadata_scope=metadata_scope)
                previous_plan = (
                    previous["head"]
                    .get("addons", {})
                    .get("euler_transforms", {})
                    .get("plan")
                )
                if self._plan is None or previous_plan != self._plan.to_dict():
                    raise ValueError(
                        "resume output plan/revision conflicts with completed output"
                    )
                self._records = validate_output_records(
                    self._root, previous, metadata_scope=metadata_scope
                )
                self._dataset_node = deepcopy(previous["index"])
                self._committed = dict(iter_entries(self._dataset_node))
                self._count = len(self._committed)

    @property
    def output_plan(self):
        return self._plan

    @property
    def destination(self):
        return self._destination

    def set_head_addon(self, name, payload):
        if self._count or name in {
            "euler_transforms",
            "euler_loading",
            "euler_representation",
        }:
            raise ValueError("validated output head is frozen")
        super().set_head_addon(name, payload)

    def get_path(self, *args, **kwargs):
        raise ValueError(
            "validated writers require commit_bytes after successful encoding"
        )

    def commit_bytes(self, full_id, basename, data, *, attributes=None, records=None):
        if self._finished:
            raise RuntimeError("ZIP writer is finalized/closed")
        check(full_id, FULL_ID, "output full_id")
        if Path(basename).name != basename or not basename or "\\" in basename:
            raise ValueError("basename must be a filename")
        if self._expected_ids is not None and full_id not in self._expected_ids:
            raise ValueError("output_full_id is outside the declared expected set")
        attributes = deepcopy(attributes or {})
        records = deepcopy(records or {})
        if self._plan is not None:
            location = attributes.get("euler_transforms")
            receipt = resolve_receipt(
                location, lambda p: records[p] if p in records else self._records[p]
            )
            validate_artifact_receipt(receipt, self._plan, full_id=full_id)
            if bytes_digest(data) != receipt["artifact"]["digest"]:
                raise ValueError("committed bytes disagree with artifact receipt")
        path = f"artifacts/{bytes_digest(data)[7:]}{Path(basename).suffix.lower()}"
        previous = self._committed.get(full_id)
        if previous:
            if previous["path"] != path or previous.get("attributes", {}) != attributes:
                raise ValueError("conflicting rewrite for qualified output ID")
            if read_artifact(self._root, path) != data:
                raise ValueError("existing output bytes changed")
            return (
                str(self._destination) + "::" + path
                if self._zip_output
                else str(self._root / path)
            )
        for key, value in records.items():
            _safe_relative(key)
            if key != f"records/{canonical_digest(value)[7:]}.json":
                raise ValueError("record path/content digest mismatch")
            if key in self._records and self._records[key] != value:
                raise ValueError("contradictory referenced record")
        _atomic_bytes(self._root / path, data)
        # Entry publication is last; encoding and byte storage have succeeded.
        _, hierarchy = self._register_entry(full_id, basename, attributes=attributes)
        node = self._dataset_node
        for key in hierarchy:
            node = node["children"][key]
        entry = node["files"][-1]
        entry["path"] = path
        self._committed[full_id] = entry
        self._records.update(records)
        return (
            str(self._destination) + "::" + path
            if self._zip_output
            else str(self._root / path)
        )

    def build_output(self):
        output = deepcopy(super().build_output())
        if self._plan is not None:
            output["head"].setdefault("addons", {})["euler_transforms"] = {
                "version": "2.0",
                "state": "materialized",
                "required_features": [],
                "plan": self._plan.to_dict(),
                "plan_digest": canonical_digest(self._plan.to_dict()),
                "mapping": {
                    "mode": "explicit",
                    "count": len(self),
                    "receipts": "per_file",
                    "index_digest": canonical_digest(output["index"]),
                },
            }
        return output

    def finalize(self):
        if self._finished:
            raise RuntimeError("ZIP writer is finalized/closed")
        output = self.build_output()
        if self._publication_validator is not None:
            self._publication_validator(
                output,
                lambda path: (
                    self._records[path]
                    if path in self._records
                    else read_record(
                        self._root, path, metadata_scope=self._metadata_scope
                    )
                ),
                lambda path: read_artifact(self._root, path),
            )
        path = finalize_output(
            self._root,
            output,
            records=self._records,
            metadata_scope=self._metadata_scope,
            expected_full_ids=self._expected_ids,
        )
        if self._zip_output:
            self._destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(
                prefix=".pending-", dir=self._destination.parent
            )
            os.close(fd)
            try:
                with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file in sorted(self._root.rglob("*")):
                        if file.is_file():
                            archive.write(file, file.relative_to(self._root).as_posix())
                os.replace(temp, self._destination)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            self._finished = True
            self._staging.cleanup()
            return self._destination
        return path

    def save_index(self, filename=OUTPUT_FILENAME):
        if filename != OUTPUT_FILENAME:
            raise ValueError("validated publication uses index.json")
        return self.finalize()

    def close(self):
        if self._staging is not None:
            self._finished = True
            self._staging.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def copy_materialized_dataset(
    source, target, output, *, sample=None, input_scope=None, output_scope=None
):
    """Preserve verified selected mappings and their authoritative records."""
    # Validate the selected subset against the original head; copy/split derives
    # a new membership digest while preserving qualified artifact identities.
    records = validate_output_records(
        source, output, metadata_scope=input_scope, subset=True
    )
    entries = list(iter_entries(output["index"]))
    if sample is not None and sample > 1:
        entries = sorted(entries, key=lambda item: item[0])[::sample]
    plan = output["head"]["addons"]["euler_transforms"]["plan"]
    with ValidatedDatasetWriter(
        target,
        head=output["head"],
        plan=plan,
        expected_full_ids=[fid for fid, _ in entries],
        metadata_scope=output_scope,
        separator=output.get("indexing", {}).get("hierarchy", {}).get("separator"),
    ) as writer:
        for full_id, entry in entries:
            writer.commit_bytes(
                full_id,
                Path(entry["path"]).name,
                read_artifact(source, entry["path"]),
                attributes=entry.get("attributes"),
                records=records,
            )
        writer.finalize()
    return {"copied": len(entries), "missing": 0, "missing_files": []}
