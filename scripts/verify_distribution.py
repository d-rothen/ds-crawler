"""Reject unexpected or unsafe files in built release archives."""

from __future__ import annotations

import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

SDIST_FILES = {
    ".gitignore",
    "CONTRIBUTING.md",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
}
SDIST_DIRECTORIES = {"docs", "ds_crawler", "examples", "meta", "tests"}
WHEEL_DIRECTORIES = {"ds_crawler", "meta"}


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return tuple(part for part in path.parts if part not in {"", "."})


def _verify_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {_safe_parts(member.name)[0] for member in members}
        if len(roots) != 1:
            raise ValueError(f"Expected one source root in {path}, found {roots}")

        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in {path}: {member.name}")
            parts = _safe_parts(member.name)
            if len(parts) == 1:
                continue
            relative = parts[1:]
            if relative[0] in SDIST_DIRECTORIES:
                continue
            if len(relative) == 1 and relative[0] in SDIST_FILES:
                continue
            raise ValueError(f"Unexpected source distribution file: {member.name}")


def _verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            parts = _safe_parts(info.filename)
            if not parts:
                continue
            file_type = (info.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(file_type):
                raise ValueError(f"Links are not allowed in {path}: {info.filename}")
            if parts[0] in WHEEL_DIRECTORIES or parts[0].endswith(".dist-info"):
                continue
            raise ValueError(f"Unexpected wheel file: {info.filename}")


def main(paths: list[str]) -> None:
    if not paths:
        raise ValueError("Pass at least one .tar.gz or .whl archive")

    for raw_path in paths:
        path = Path(raw_path)
        if path.name.endswith(".tar.gz"):
            _verify_sdist(path)
        elif path.suffix == ".whl":
            _verify_wheel(path)
        else:
            raise ValueError(f"Unsupported distribution archive: {path}")
        print(f"Verified {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
