"""Dataset handlers package."""

from .base import BaseHandler
from .vkitti2 import VKITTI2Handler

HANDLERS = {
    "VKITTI2": VKITTI2Handler,
}


def get_handler(name: str) -> type[BaseHandler]:
    """Get handler class by dataset name."""
    if name not in HANDLERS:
        raise ValueError(f"Unknown dataset name: {name}. Available: {list(HANDLERS.keys())}")
    return HANDLERS[name]


__all__ = ["BaseHandler", "VKITTI2Handler", "get_handler", "HANDLERS"]
