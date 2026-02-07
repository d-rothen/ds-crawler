"""Dataset writer for constructing output directories and index files.

Provides :class:`DatasetWriter`, a stateful helper that turns
``(full_id, basename)`` pairs into filesystem paths while accumulating
the metadata needed to produce a valid ``output.json`` — the same format
that :func:`~ds_crawler.parser.index_dataset_from_path` returns.

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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import _MODALITY_META_SCHEMAS
from .zip_utils import write_metadata_json

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


class DatasetWriter:
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
            for key, (expected_type, type_label, _desc) in schema.items():
                if key not in meta:
                    raise ValueError(
                        f"meta.{key} is required for "
                        f"modality_type={modality_type!r}"
                    )
                if not isinstance(meta[key], expected_type):
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
        """The output root directory."""
        return self._root

    def __len__(self) -> int:
        """Number of file entries registered so far."""
        return self._count

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

        # --- Compute paths ----------------------------------------------
        rel_dir = Path(*dir_parts) if dir_parts else Path()
        rel_path = rel_dir / basename
        abs_path = self._root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

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

        return abs_path

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
