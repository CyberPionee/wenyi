"""Disposable target text for in-app reading.

Storage keeps the formal translation exactly as it was produced, and the reading view must show
the same text an export would contain. This module therefore applies the selected target-language
text operations to a detached copy; it never touches a chapter, a manifest or any other state.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from ..i18n.policy.models import ExportTextInput, PolicyContext
from ..i18n.policy.resolver import resolve_policy
from ..postprocess.export_text import apply_operations, logical_ranges
from .export_view import TEXT_HANDLERS


@lru_cache(maxsize=32)
def _text_operations(
    source: str,
    target: str,
    punctuation_normalize: bool,
) -> tuple[str, ...]:
    """Report the export text operations this language direction applies.

    The reading view renders HTML-shaped text, so it resolves the export plan with the HTML
    format. Nothing here is persisted, so the resolved fingerprint is discarded.
    """
    plan = resolve_policy(
        PolicyContext(
            source,
            target,
            phase="export",
            format="html",
            punctuation_normalize=punctuation_normalize,
        )
    )
    return tuple(operation.id for operation in plan.operations if operation.point == "export.text")


def display_targets(
    segment_ids: Sequence[int],
    targets: Sequence[str | None],
    continuations: Sequence[bool],
    *,
    source: str,
    target: str,
    punctuation_normalize: bool = True,
) -> list[str | None]:
    """Return a display copy of the target texts for one chapter.

    Segment identities, order, continuation boundaries, pending ``None`` and deliberately empty
    ``""`` targets keep their meaning, so callers can address the returned row exactly as they
    address the stored one.
    """
    values = list(targets)
    operations = _text_operations(source, target, punctuation_normalize)
    if not operations:
        return values
    request = ExportTextInput(
        tuple(segment_ids),
        tuple(values),
        tuple(continuations),
        logical_ranges(continuations),
    )
    result, _maps = apply_operations(request, TEXT_HANDLERS, operations)
    return list(result.targets)
