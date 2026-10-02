"""Tolerant parsing of model JSON output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from json_repair import repair_json


class JsonParseError(ValueError):
    """Model output remains unusable JSON after local repair."""


@dataclass(frozen=True)
class JsonParseResult:
    """Parsed JSON result, whether repair was needed, and how invasive it was.

    ``repair_kind`` classifies the repair so callers can refuse to trust a payload
    that could only be recovered by inventing missing structure:

    - ``none``: strict ``json.loads`` succeeded.
    - ``boundary_only``: only an outer boundary had to be completed or trimmed
      (one appended root closer, surplus trailing closers, or an unwrapped fence).
      The payload itself was already whole.
    - ``structural``: strings, collections or values had to be synthesized, so the
      result may be a truncated fragment rather than the model's full answer.
    """

    value: Any
    repaired: bool
    repair_kind: Literal["none", "boundary_only", "structural"] = "none"

    @property
    def safe_for_complete_payload(self) -> bool:
        """True when the payload counts as the model's complete output."""
        return self.repair_kind in {"none", "boundary_only"}


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare two parsed values without bool/int coercion."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _strict_loads(text: str) -> Any:
    return json.loads(text)


def _is_boundary_only_repair(raw: str, value: Any) -> bool:
    """True when appending/stripping outer closers or unwrapping a fence suffices."""
    if value == "":
        return False
    candidates: list[str] = []
    stripped = raw.rstrip()
    for closer in ("}", "]"):
        candidates.append(stripped + closer)
    # Surplus trailing closers, e.g. "{...}}".
    trimmed = stripped
    while trimmed.endswith(("}", "]")):
        trimmed = trimmed[:-1]
        if trimmed:
            candidates.append(trimmed)
    # Markdown fence unwrapping.
    if "```" in raw:
        fenced = raw
        if fenced.startswith("```"):
            first_newline = fenced.find("\n")
            if first_newline != -1:
                fenced = fenced[first_newline + 1 :]
        elif "```" in fenced:
            fenced = fenced[fenced.index("```") + 3 :]
            if fenced.startswith("json"):
                fenced = fenced[4:]
            if fenced.startswith("\n"):
                fenced = fenced[1:]
        if fenced.endswith("```"):
            fenced = fenced[:-3]
        candidates.append(fenced.strip())
    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            parsed = _strict_loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if _same_json_value(parsed, value):
            return True
    return False


def parse_json_result(text: str) -> JsonParseResult:
    """Parse model JSON and classify how invasive the repair was.

    Run json.loads once to determine the repaired flag. On failure, call json-repair
    with skip_json_loads=True to avoid repeating strict parsing.
    """
    raw = (text or "").strip()
    try:
        return JsonParseResult(json.loads(raw), repaired=False, repair_kind="none")
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        value = repair_json(
            raw,
            return_objects=True,
            skip_json_loads=True,
        )
    except Exception as error:
        raise JsonParseError(f"Cannot parse JSON: {raw[:200]!r}") from error
    # json-repair returns an empty string for empty/plain-language input; that is not recoverable JSON.
    if value == "":
        raise JsonParseError(f"Cannot parse JSON: {raw[:200]!r}")
    kind: Literal["boundary_only", "structural"] = (
        "boundary_only" if _is_boundary_only_repair(raw, value) else "structural"
    )
    return JsonParseResult(value, repaired=True, repair_kind=kind)


def parse_json_loose(text: str) -> Any:
    """Return the parsed value, delegating syntax recovery to json-repair."""
    return parse_json_result(text).value
