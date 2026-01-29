"""Dataset handlers package."""

from __future__ import annotations

from pathlib import Path

from .base import BaseHandler
from .generic import GenericHandler
from .zip_handler import ZipHandler

# Registry for dataset-specific handlers. Datasets without a registered
# handler automatically use GenericHandler.
HANDLERS: dict[str, type[BaseHandler]] = {}


def get_handler(
    name: str, path: str | Path | None = None
) -> type[BaseHandler]:
    """Get handler class by dataset name.

    If the dataset has a registered handler it is returned regardless of
    *path*.  Otherwise, when *path* points to a ``.zip`` file the
    :class:`ZipHandler` is returned; in all other cases the default
    :class:`GenericHandler` is used.
    """
    if name in HANDLERS:
        return HANDLERS[name]
    if path is not None:
        from ..zip_utils import is_zip_path

        if is_zip_path(path):
            return ZipHandler
    return GenericHandler


__all__ = ["BaseHandler", "GenericHandler", "ZipHandler", "get_handler", "HANDLERS"]
