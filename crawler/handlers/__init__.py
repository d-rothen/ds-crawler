"""Dataset handlers package."""

from .base import BaseHandler
from .generic import GenericHandler

# Registry for dataset-specific handlers. Datasets without a registered
# handler automatically use GenericHandler.
HANDLERS: dict[str, type[BaseHandler]] = {}


def get_handler(name: str) -> type[BaseHandler]:
    """Get handler class by dataset name.

    Returns GenericHandler for datasets without a registered handler.
    """
    return HANDLERS.get(name, GenericHandler)


__all__ = ["BaseHandler", "GenericHandler", "get_handler", "HANDLERS"]
