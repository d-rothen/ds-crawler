"""Dataset writers for constructing output directories/archives and index files.

Provides :class:`DatasetWriter` (filesystem) and :class:`ZipDatasetWriter`
(ZIP archive), both backed by :class:`_BaseDatasetWriter` which handles
index accumulation and ``output.json`` generation.

Typical usage with ``euler-loading``::

    from ds_crawler import DatasetWriter

    writer = DatasetWriter(
        "/output/segmentation",
        name="segmentation",
        type="segmentation",
        euler_train={"used_as": "target", "modality_type": "semantic"},
        separator=":",
    )

    for sample in dataloader:
        prediction = model(sample["rgb"], sample["depth"])
        out_path = writer.get_path(
            full_id=sample["full_id"],
            basename=f"{sample['id']}.png",
        )
        save_image(prediction, out_path)

    writer.save_index()  # writes .ds_crawler/output.json

ZIP output::

    from ds_crawler import ZipDatasetWriter

    with ZipDatasetWriter(
        "/output/segmentation.zip",
        name="segmentation",
        type="segmentation",
        euler_train={"used_as": "target", "modality_type": "semantic"},
        separator=":",
    ) as writer:
        for sample in dataloader:
            with writer.open(
                full_id=sample["full_id"],
                basename=f"{sample['id']}.png",
            ) as f:
                save_image(prediction, f)

        writer.save_index()
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from .config import _MODALITY_META_SCHEMAS
from .zip_utils import COMPRESSED_EXTENSIONS, METADATA_DIR, write_metadata_json

logger = logging.getLogger(__name__)


def _parse_full_id(full_id: str) -> tuple[list[str], str]:
    """Split a ``full_id`` into hierarchy keys and a leaf file ID.

    ``full_id`` has the form ``"/key1/key2/.../leaf_id"`` (leading slash
    optional).  Returns ``(["key1", "key2", ...], "leaf_id")``.

    Raises :class:`ValueError` when *full_id* contains no usable parts.
    """
    parts = [p for p in full_id.split("/") if p]
    if not parts:
        raise ValueError(f"full_id is empty or contains only slashes: {full_id!r}")
    return parts[:-1], parts[-1]


def _split_hierarchy_key(
    key: str, separator: str | None
) -> tuple[str | None, str]:
    """Split a hierarchy key into ``(name, value)``.

    When *separator* is set and present in *key*:
    ``"scene:Scene01"`` → ``("scene", "Scene01")``.

    Otherwise returns ``(None, key)`` (unnamed / no separator).
    """
    if separator and separator in key:
        name, value = key.split(separator, 1)
        return name, value
    return None, key


# ======================================================================
# Base class — shared index-building logic
# ======================================================================


class _BaseDatasetWriter:
    """Shared logic for accumulating file entries and building ``output.json``.

    Not intended for direct use.  Subclass and provide a storage-specific
    write interface (filesystem paths or ZIP entries).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        type: str,
        euler_train: dict[str, Any],
        separator: str | None = ":",
        **properties: Any,
    ) -> None:
        if "used_as" not in euler_train:
            raise ValueError("euler_train must contain 'used_as'")
        if "modality_type" not in euler_train:
            raise ValueError("euler_train must contain 'modality_type'")

        modality_type = euler_train["modality_type"]
        schema = _MODALITY_META_SCHEMAS.get(modality_type)
        if schema is not None:
            meta = properties.get("meta")
            if meta is None or not isinstance(meta, dict):
                required_keys = ", ".join(sorted(schema))
                raise ValueError(
                    f"meta is required for modality_type={modality_type!r} "
                    f"and must contain: {required_keys}"
                )
            for key, entry in schema.items():
                expected_type, type_label, _desc = entry[:3]
                validator = entry[3] if len(entry) > 3 else None
                if key not in meta:
                    raise ValueError(
                        f"meta.{key} is required for "
                        f"modality_type={modality_type!r}"
                    )
                value = meta[key]
                if validator is not None:
                    err = validator(value)
                    if err is not None:
                        raise ValueError(f"meta.{key} must be {err}")
                elif not isinstance(value, expected_type):
                    raise ValueError(f"meta.{key} must be {type_label}")

        self._root = Path(root)
        self._name = name
        self._type = type
        self._euler_train = dict(euler_train)
        self._separator = separator
        self._properties = properties
        self._dataset_node: dict[str, Any] = {}
        self._count = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The output root (directory or ``.zip`` path)."""
        return self._root

    def __len__(self) -> int:
        """Number of file entries registered so far."""
        return self._count

    # ------------------------------------------------------------------
    # Entry registration (shared)
    # ------------------------------------------------------------------

    def _register_entry(
        self,
        full_id: str,
        basename: str,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        """Parse *full_id*, record the entry, return ``(rel_path_str, hierarchy_keys)``.

        This is the shared core of both :meth:`DatasetWriter.get_path` and
        :meth:`ZipDatasetWriter.open` / :meth:`ZipDatasetWriter.write`.
        """
        hierarchy_keys, file_id = _parse_full_id(full_id)

        # --- Build directory segments and path_properties ---------------
        dir_parts: list[str] = []
        path_properties: dict[str, str] = {}

        for key in hierarchy_keys:
            name, value = _split_hierarchy_key(key, self._separator)
            dir_parts.append(value)
            if name is not None:
                path_properties[name] = value

        # --- Override properties from source if provided ----------------
        if source_meta is not None:
            if "path_properties" in source_meta:
                path_properties = dict(source_meta["path_properties"])

        # --- Compute relative path --------------------------------------
        rel_dir = Path(*dir_parts) if dir_parts else Path()
        rel_path = rel_dir / basename

        # --- Build basename_properties ----------------------------------
        if source_meta is not None and "basename_properties" in source_meta:
            basename_properties: dict[str, str] = dict(
                source_meta["basename_properties"]
            )
        else:
            basename_properties = {}
            suffix = Path(basename).suffix
            if suffix:
                basename_properties["ext"] = suffix.lstrip(".")

        # --- Build file entry -------------------------------------------
        entry: dict[str, Any] = {
            "path": str(rel_path),
            "id": file_id,
            "path_properties": path_properties,
            "basename_properties": basename_properties,
        }

        # --- Place in hierarchy -----------------------------------------
        node = self._dataset_node
        for key in hierarchy_keys:
            if "children" not in node:
                node["children"] = {}
            if key not in node["children"]:
                node["children"][key] = {}
            node = node["children"][key]

        if "files" not in node:
            node["files"] = []
        node["files"].append(entry)
        self._count += 1

        return str(rel_path), hierarchy_keys

    # ------------------------------------------------------------------
    # Index output
    # ------------------------------------------------------------------

    def build_output(self) -> dict[str, Any]:
        """Return the accumulated index as an output dict.

        The returned dict has the same structure as the output of
        :func:`~ds_crawler.parser.index_dataset_from_path` and can be
        passed directly to :func:`~ds_crawler.parser.align_datasets`.
        """
        output: dict[str, Any] = {
            "name": self._name,
            "type": self._type,
            "euler_train": self._euler_train,
            **self._properties,
            "dataset": self._dataset_node,
        }
        if self._separator is not None:
            output["named_capture_group_value_separator"] = self._separator
        return output


# ======================================================================
# Filesystem writer
# ======================================================================


class DatasetWriter(_BaseDatasetWriter):
    """Accumulates file entries and produces ``output.json``-compatible output.

    Each call to :meth:`get_path` registers a file entry and returns the
    absolute path where the caller should write the actual data.  When all
    files have been written, :meth:`save_index` persists the accumulated
    index as ``.ds_crawler/output.json`` — ready for consumption by
    :func:`~ds_crawler.parser.index_dataset_from_path`,
    :func:`~ds_crawler.parser.align_datasets`, or
    ``euler_loading.MultiModalDataset``.

    Args:
        root: Output directory.  Subdirectories are created as needed.
        name: Dataset name (written to the index).
        type: Semantic label for the data modality (e.g. ``"rgb"``,
            ``"depth"``).
        euler_train: Training metadata dict.  Must contain at least
            ``used_as`` and ``modality_type``.
        separator: The character used to join hierarchy key names and
            values (e.g. ``":"`` for ``"scene:Scene01"``).  Should match
            the ``named_capture_group_value_separator`` of the source
            datasets.  Pass ``None`` for unnamed hierarchy keys.
        **properties: Arbitrary extra metadata written to the index
            (e.g. ``gt=False``, ``model="MyModel"``).
    """

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def get_path(
        self,
        full_id: str,
        basename: str,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Register a file entry and return the absolute path to write to.

        Parses *full_id* (e.g. ``"/scene:Scene01/camera:Cam0/00001"``)
        into hierarchy keys + leaf ID, builds the directory tree under
        :attr:`root`, and records the entry for later :meth:`save_index`.

        Args:
            full_id: Hierarchical identifier as produced by
                ``euler_loading.MultiModalDataset`` (``sample["full_id"]``).
                Leading slash is optional.
            basename: Filename including extension (e.g. ``"00001.png"``).
            source_meta: Optional source file entry dict (e.g.
                ``sample["meta"]["rgb"]``).  When provided, its
                ``path_properties`` and ``basename_properties`` are
                copied verbatim into the new entry instead of being
                reconstructed from the hierarchy keys.

        Returns:
            Absolute :class:`~pathlib.Path` to the file.  Parent
            directories are created automatically.
        """
        rel_path_str, _hierarchy_keys = self._register_entry(
            full_id, basename, source_meta=source_meta,
        )
        abs_path = self._root / rel_path_str
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path

    # ------------------------------------------------------------------
    # Index output
    # ------------------------------------------------------------------

    def save_index(self, filename: str = "output.json") -> Path:
        """Write the accumulated index to ``.ds_crawler/{filename}``.

        Args:
            filename: Name of the JSON file inside the ``.ds_crawler``
                directory.

        Returns:
            The :class:`~pathlib.Path` that was written to.
        """
        output = self.build_output()
        path = write_metadata_json(self._root, filename, output)
        logger.info(
            "DatasetWriter: saved index with %d entries to %s",
            self._count,
            path,
        )
        return path


# ======================================================================
# ZIP writer
# ======================================================================


class _ZipEntryFile(io.BytesIO):
    """A writable file-like that flushes its contents into a ZIP on close."""

    def __init__(self, zf: zipfile.ZipFile, entry_name: str, compress_type: int):
        super().__init__()
        self._zf = zf
        self._entry_name = entry_name
        self._compress_type = compress_type
        self._flushed = False

    def close(self) -> None:
        if not self._flushed:
            self._zf.writestr(
                self._entry_name,
                self.getvalue(),
                compress_type=self._compress_type,
            )
            self._flushed = True
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ZipDatasetWriter(_BaseDatasetWriter):
    """Write dataset files directly into a ``.zip`` archive.

    Unlike :class:`DatasetWriter`, which writes to the filesystem,
    this class keeps a :class:`zipfile.ZipFile` open and provides
    :meth:`open` (returns a writable file-like) and :meth:`write`
    (accepts raw bytes) for storing data entries.

    Use as a context manager to ensure the archive is closed even
    when :meth:`save_index` is not called::

        with ZipDatasetWriter("output.zip", ...) as writer:
            with writer.open(full_id, basename) as f:
                np.save(f, array)
            writer.save_index()

    Args:
        root: Output ``.zip`` file path.  Parent directories are
            created if needed.  The file must not already exist.
        name: Dataset name (written to the index).
        type: Semantic label for the data modality.
        euler_train: Training metadata dict (must contain ``used_as``
            and ``modality_type``).
        separator: Hierarchy key separator (see :class:`DatasetWriter`).
        **properties: Extra metadata written to the index.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        name: str,
        type: str,
        euler_train: dict[str, Any],
        separator: str | None = ":",
        **properties: Any,
    ) -> None:
        super().__init__(
            root, name=name, type=type, euler_train=euler_train,
            separator=separator, **properties,
        )
        if self._root.suffix.lower() != ".zip":
            raise ValueError(
                f"ZipDatasetWriter root must be a .zip path, got: {self._root}"
            )
        self._root.parent.mkdir(parents=True, exist_ok=True)
        self._zf = zipfile.ZipFile(self._root, "w", zipfile.ZIP_STORED)
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _compress_type_for(self, basename: str) -> int:
        suffix = Path(basename).suffix.lower()
        if suffix in COMPRESSED_EXTENSIONS:
            return zipfile.ZIP_STORED
        return zipfile.ZIP_DEFLATED

    def open(
        self,
        full_id: str,
        basename: str,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> _ZipEntryFile:
        """Register a file entry and return a writable file-like object.

        Data written to the returned object is flushed into the ZIP
        archive when the object is closed.  Works with any API that
        accepts a file-like (``np.save``, ``PIL.Image.save``, etc.)::

            with writer.open(full_id, basename) as f:
                np.save(f, array)

        Args:
            full_id: Hierarchical identifier (see :meth:`DatasetWriter.get_path`).
            basename: Filename including extension.
            source_meta: Optional source file entry dict.

        Returns:
            A writable :class:`io.BytesIO` subclass that writes into
            the ZIP on ``.close()``.
        """
        if self._closed:
            raise RuntimeError("ZipDatasetWriter is closed")
        rel_path_str, _ = self._register_entry(
            full_id, basename, source_meta=source_meta,
        )
        entry_name = rel_path_str.replace("\\", "/")
        compress = self._compress_type_for(basename)
        return _ZipEntryFile(self._zf, entry_name, compress)

    def write(
        self,
        full_id: str,
        basename: str,
        data: bytes,
        *,
        source_meta: dict[str, Any] | None = None,
    ) -> None:
        """Register a file entry and write raw bytes into the archive.

        Convenience method for when the data is already available as
        bytes.  For streaming writes, use :meth:`open` instead.

        Args:
            full_id: Hierarchical identifier.
            basename: Filename including extension.
            data: Raw file contents.
            source_meta: Optional source file entry dict.
        """
        if self._closed:
            raise RuntimeError("ZipDatasetWriter is closed")
        rel_path_str, _ = self._register_entry(
            full_id, basename, source_meta=source_meta,
        )
        entry_name = rel_path_str.replace("\\", "/")
        compress = self._compress_type_for(basename)
        self._zf.writestr(entry_name, data, compress_type=compress)

    # ------------------------------------------------------------------
    # Index output
    # ------------------------------------------------------------------

    def save_index(self, filename: str = "output.json") -> Path:
        """Write the index into the archive and close it.

        Args:
            filename: Name of the JSON metadata file.

        Returns:
            The ``.zip`` :class:`~pathlib.Path`.
        """
        if self._closed:
            raise RuntimeError("ZipDatasetWriter is closed")
        output = self.build_output()
        self._zf.writestr(
            f"{METADATA_DIR}/{filename}",
            json.dumps(output, indent=2),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        self._zf.close()
        self._closed = True
        logger.info(
            "ZipDatasetWriter: saved index with %d entries to %s",
            self._count,
            self._root,
        )
        return self._root

    def close(self) -> None:
        """Close the underlying ZIP archive (idempotent)."""
        if not self._closed:
            self._zf.close()
            self._closed = True
