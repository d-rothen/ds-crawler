"""Package version helper."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_package_version() -> str:
    """Return the installed ds_crawler package version, or ``"0"`` as fallback."""
    try:
        return version("ds_crawler")
    except PackageNotFoundError:
        return "0"
