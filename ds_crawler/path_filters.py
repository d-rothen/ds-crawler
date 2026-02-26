"""Path-based include/exclude filtering for dataset entries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_ALLOWED_KEYS: frozenset[str] = frozenset({
    "include_regex",
    "exclude_regex",
    "include_terms",
    "exclude_terms",
    "term_match_mode",
    "case_sensitive",
})
_ALLOWED_TERM_MATCH_MODES: frozenset[str] = frozenset({
    "substring",
    "path_segment",
})


@dataclass
class PathFilters:
    """Normalized include/exclude path filters.

    Filters are evaluated in this order:
    1) include_regex (at least one must match when configured),
    2) exclude_regex (none may match),
    3) include_terms (at least one must match when configured),
    4) exclude_terms (none may match).
    """

    include_regex: tuple[str, ...] = ()
    exclude_regex: tuple[str, ...] = ()
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    term_match_mode: str = "substring"
    case_sensitive: bool = True
    _compiled_include_regex: tuple[re.Pattern, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    _compiled_exclude_regex: tuple[re.Pattern, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    _normalized_include_terms: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    _normalized_exclude_terms: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )

    @classmethod
    def from_raw(cls, raw: Any, *, context: str = "path_filters") -> "PathFilters":
        """Create filters from a raw config object."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")

        unknown = sorted(set(raw.keys()) - _ALLOWED_KEYS)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"Unknown {context} key(s): {joined}")

        term_match_mode = raw.get("term_match_mode", "substring")
        if term_match_mode not in _ALLOWED_TERM_MATCH_MODES:
            allowed = ", ".join(sorted(_ALLOWED_TERM_MATCH_MODES))
            raise ValueError(
                f"{context}.term_match_mode must be one of {{{allowed}}}, "
                f"got {term_match_mode!r}"
            )

        case_sensitive = raw.get("case_sensitive", True)
        if not isinstance(case_sensitive, bool):
            raise ValueError(f"{context}.case_sensitive must be a boolean")

        include_regex = _normalize_string_list(
            raw.get("include_regex"),
            f"{context}.include_regex",
        )
        exclude_regex = _normalize_string_list(
            raw.get("exclude_regex"),
            f"{context}.exclude_regex",
        )
        include_terms = _normalize_string_list(
            raw.get("include_terms"),
            f"{context}.include_terms",
        )
        exclude_terms = _normalize_string_list(
            raw.get("exclude_terms"),
            f"{context}.exclude_terms",
        )

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_include_regex = tuple(
            _compile_regex(expr, f"{context}.include_regex[{idx}]", flags)
            for idx, expr in enumerate(include_regex)
        )
        compiled_exclude_regex = tuple(
            _compile_regex(expr, f"{context}.exclude_regex[{idx}]", flags)
            for idx, expr in enumerate(exclude_regex)
        )

        normalized_include_terms = include_terms
        normalized_exclude_terms = exclude_terms
        if not case_sensitive:
            normalized_include_terms = tuple(term.lower() for term in include_terms)
            normalized_exclude_terms = tuple(term.lower() for term in exclude_terms)

        return cls(
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
            term_match_mode=term_match_mode,
            case_sensitive=case_sensitive,
            _compiled_include_regex=compiled_include_regex,
            _compiled_exclude_regex=compiled_exclude_regex,
            _normalized_include_terms=normalized_include_terms,
            _normalized_exclude_terms=normalized_exclude_terms,
        )

    @property
    def enabled(self) -> bool:
        """Return whether any include/exclude rule is configured."""
        return bool(
            self.include_regex
            or self.exclude_regex
            or self.include_terms
            or self.exclude_terms
        )

    def to_dict(self) -> dict[str, Any]:
        """Return normalized dict form for serialization/output metadata."""
        if not self.enabled:
            return {}

        result: dict[str, Any] = {}
        if self.include_regex:
            result["include_regex"] = list(self.include_regex)
        if self.exclude_regex:
            result["exclude_regex"] = list(self.exclude_regex)
        if self.include_terms:
            result["include_terms"] = list(self.include_terms)
        if self.exclude_terms:
            result["exclude_terms"] = list(self.exclude_terms)
        if self.term_match_mode != "substring":
            result["term_match_mode"] = self.term_match_mode
        if not self.case_sensitive:
            result["case_sensitive"] = False
        return result

    def matches(self, path: str) -> bool:
        """Return True when *path* passes all include/exclude checks."""
        if not self.enabled:
            return True

        normalized_path = path.replace("\\", "/")
        candidate = normalized_path if self.case_sensitive else normalized_path.lower()

        if self._compiled_include_regex and not any(
            regex.search(normalized_path) for regex in self._compiled_include_regex
        ):
            return False
        if self._compiled_exclude_regex and any(
            regex.search(normalized_path) for regex in self._compiled_exclude_regex
        ):
            return False

        segments: tuple[str, ...] | None = None
        if self.term_match_mode == "path_segment":
            segments = tuple(seg for seg in candidate.split("/") if seg)

        if self._normalized_include_terms and not _terms_match(
            self._normalized_include_terms,
            candidate,
            segments,
            self.term_match_mode,
        ):
            return False
        if self._normalized_exclude_terms and _terms_match(
            self._normalized_exclude_terms,
            candidate,
            segments,
            self.term_match_mode,
        ):
            return False

        return True


def _normalize_string_list(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of non-empty strings")

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        text = item.strip()
        if not text:
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(text)
    return tuple(result)


def _compile_regex(pattern: str, label: str, flags: int) -> re.Pattern:
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"{label} is not a valid regex: {exc}") from exc


def _terms_match(
    terms: tuple[str, ...],
    candidate: str,
    segments: tuple[str, ...] | None,
    mode: str,
) -> bool:
    if mode == "substring":
        return any(term in candidate for term in terms)
    return any(term in (segments or ()) for term in terms)
