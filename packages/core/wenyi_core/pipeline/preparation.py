"""Preparation: state lookup, parsing, language detection, initialization, analysis and
prescan.
Own PDF conversion caches, source hashes, sample selection, initial glossary and rolling
context. Initialize derived chapters/analysis/glossary/context first, atomically commit the
initialized manifest last, then finish initialization. Build chapter digests and the book
synopsis as configured. Share pure language normalization with Runtime through top-level i18n.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from ..i18n.languages import normalize_language
from ..i18n.policy.models import Phase, content_hash
from ..i18n.prompts import render
from ..ingest.epub_reader import peek_epub_title
from ..ingest.models import Document
from ..ingest.segmenter import load_document
from ..storage.protocol import Storage
from .context import RollingContext
from .language_policies import commit_revision, initialize_policies, translation_revision
from .runstore import source_sha256, translation_run_dir

if TYPE_CHECKING:
    from .runtime import PipelineRuntime

ProgressFn = Callable[[int, int, str], None]


def _synopsis_complete(text: str) -> bool:
    """A usable whole-book synopsis must be non-empty and end like finished prose."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return cleaned[-1] in "。．.！？!?…”\"'」』）)]}"


class PreparationService:
    """Domain service for state lookup, parsing, initialization and book understanding."""

    def __init__(self, runtime: PipelineRuntime):
        self._runtime = runtime

    @staticmethod
    def ingest_config(config) -> dict[str, Any]:
        """Fingerprint only parsing inputs shared by upload preview and initialization."""
        return {
            "source_lang": config.source_lang,
            "target_lang": config.target_lang,
            "max_tokens_per_segment": config.segment.max_tokens_per_segment,
            "pdf_backend": config.pipeline.pdf_backend,
            "babeldoc_bridge_url": config.pipeline.babeldoc_bridge_url,
            "babeldoc_pages": config.pipeline.babeldoc_pages,
            "babeldoc_timeout": config.pipeline.babeldoc_timeout,
        }

    @staticmethod
    def load_parsed_document(
        storage, input_path: str, config, *, actual_sha256: str | None = None
    ) -> Document | None:
        """Reuse validated preview parsing; changed input/config triggers normal parsing."""
        cached = storage.read_artifact("parsed_document.json")
        if not isinstance(cached, dict):
            return None
        digest = actual_sha256 or source_sha256(input_path)
        if cached.get("source_sha256") != digest or cached.get(
            "ingest_config"
        ) != PreparationService.ingest_config(config):
            return None
        document = Document.model_validate(cached["document"])
        document.source_path = input_path
        return document

    # State lookup and resume.
    def locate_existing(
        self,
        input_path: str,
        *,
        progress: ProgressFn | None = None,
    ) -> Storage:
        """Locate existing state without creating or initializing a translation task.
        PDF state follows the filename and can be checked before MinerU. EPUB needs only the
        OPF title, avoiding full-resource annotation that export will later repeat. Other
        formats parse their local title to match preparation's state path.
        """
        if self._runtime.storage is not None:
            store = self._runtime.storage
            if not store.exists():
                raise ValueError("No translation progress found. Run translate first.")
            self._runtime.ensure_store_source(store, input_path)
            self._runtime.bind_llm_events(store)
            return store
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf":
            title = os.path.splitext(os.path.basename(input_path))[0]
        elif ext == ".epub":
            if progress:
                progress(0, 0, "Locating translation progress…")
            title = peek_epub_title(input_path)
        else:
            if progress:
                progress(0, 0, "Locating translation progress…")
            doc = load_document(
                input_path,
                self._runtime.config.source_lang,
                self._runtime.config.target_lang,
                split_segments=self._runtime.config.segment.max_tokens_per_segment,
            )
            title = doc.title

        store = self._runtime.get_store(
            translation_run_dir(
                self._runtime.config.state_dir, title, self._runtime.config.target_lang
            ),
            create=False,
        )
        if not store.exists():
            raise ValueError("No translation progress found. Run translate first.")
        self._runtime.ensure_store_source(store, input_path)
        self._runtime.bind_llm_events(store)

        return store

    def prepare(
        self,
        input_path: str,
        *,
        progress: ProgressFn | None = None,
    ) -> Storage:
        """Parse input and locate state; initialize first runs under the book lock.
        PDF state follows the filename, allowing manifest checks before repeated external
        conversion. Cache the initial converted HTML within that state directory.
        """
        if self._runtime.storage is not None:
            store = self._runtime.storage
            self._runtime.bind_llm_events(store)
            if store.exists():
                self._runtime.ensure_store_source(store, input_path)
                return store
            digest = source_sha256(input_path)
            cached = self.load_parsed_document(
                store, input_path, self._runtime.config, actual_sha256=digest
            )
            if cached is not None:
                with store.lock():
                    return self._prepare_locked(
                        cached, store, input_path, progress, source_hash=digest
                    )
        if os.path.splitext(input_path)[1].lower() == ".pdf":
            # PDF titles use the filename, so the state directory is known before initial parsing.
            pdf_title = os.path.splitext(os.path.basename(input_path))[0]
            run_dir = translation_run_dir(
                self._runtime.config.state_dir, pdf_title, self._runtime.config.target_lang
            )
            store = self._runtime.get_store(run_dir)
            self._runtime.bind_llm_events(store)

            with store.lock():
                if store.exists():
                    self._runtime.ensure_store_source(store, input_path)
                    store.log_event(
                        "run_resumed",
                        input_path=input_path,
                        run_dir=store.run_dir,
                    )
                    return store
                if progress:
                    progress(0, 0, "Parsing document…")
                source_hash = source_sha256(input_path)
                # Preserve the source identity and event history when conversion fails.
                store.begin_initialization(source_hash)
                pipeline = self._runtime.config.pipeline
                doc = load_document(
                    input_path,
                    self._runtime.config.source_lang,
                    self._runtime.config.target_lang,
                    split_segments=self._runtime.config.segment.max_tokens_per_segment,
                    cache_dir=store.source_dir,
                    source_hash=source_hash,
                    pdf_backend=pipeline.pdf_backend,
                    babeldoc_bridge_url=pipeline.babeldoc_bridge_url,
                    babeldoc_pages=pipeline.babeldoc_pages,
                    babeldoc_timeout=pipeline.babeldoc_timeout,
                )
                if source_sha256(input_path) != source_hash:
                    raise ValueError(
                        "PDF changed during parsing; ensure the file is stable and retry."
                    )
                return self._prepare_locked(
                    doc,
                    store,
                    input_path,
                    progress,
                    source_hash=source_hash,
                )

        if progress:
            progress(0, 0, "Parsing document…")
        source_hash = source_sha256(input_path)
        # Split long paragraphs at sentences and mark continuations for later backfill merging.
        doc = load_document(
            input_path,
            self._runtime.config.source_lang,
            self._runtime.config.target_lang,
            split_segments=self._runtime.config.segment.max_tokens_per_segment,
        )
        if source_sha256(input_path) != source_hash:
            raise ValueError("Source changed during parsing; ensure the file is stable and retry.")
        run_dir = translation_run_dir(
            self._runtime.config.state_dir, doc.title, self._runtime.config.target_lang
        )
        store = self._runtime.get_store(run_dir)
        self._runtime.bind_llm_events(store)

        with store.lock():
            return self._prepare_locked(
                doc,
                store,
                input_path,
                progress,
                source_hash=source_hash,
            )

    def _prepare_locked(
        self,
        doc,
        store: Storage,
        input_path: str,
        progress: ProgressFn | None,
        *,
        source_hash: str,
    ) -> Storage:
        """Restore existing state, or write new derived state before atomically committing the
        manifest.
        """
        if store.exists():
            self._runtime.ensure_store_source(store, input_path)
            store.log_event("run_resumed", input_path=input_path, run_dir=store.run_dir)
            return store  # Resume existing progress without reset; run() restores languages from the manifest.

        store.begin_initialization(source_hash)

        # For new auto-language runs, use model detection only; require an explicit language on failure.
        if self._runtime.config.source_lang in ("auto", "", None):
            if progress:
                progress(0, 0, "Detecting language…")
            detected = self.detect_language_ai(doc)
            if not detected:
                store.log_event("language_detection_failed", source_lang=doc.source_lang)
                raise ValueError(
                    "Source language detection failed. Check model settings or set "
                    "language.source in config.yaml to a supported code, such as ja/en/zh-Hant/ko/fr/de/es."
                )
            doc.source_lang = detected
            store.log_event("language_detected", source_lang=doc.source_lang)
        self._runtime.apply_language(doc.source_lang, source_identity=source_hash)
        doc.source_lang = self._runtime.config.source_lang
        doc.target_lang = self._runtime.config.target_lang

        manifest = store.stage_document(
            doc,
            source_hash=source_hash,
        )
        manifest["language_policies"] = initialize_policies(store, self._runtime.config)
        glossary = store
        if progress:
            progress(0, 0, "Analyzing book style…")
        sample = self.sample_text(doc)
        analysis = self._runtime.analyzer.analyze(sample) if sample else {}
        analysis["language_policy"] = manifest["language_policies"]["analysis"]
        analysis["style_policy"] = self._runtime.config.language_policy(
            "analysis"
        ).task_fingerprint("analyzer")
        if analysis:
            self._runtime.analyzer.seed_glossary(glossary, analysis)
        store.save_analysis(analysis)
        store.log_event("analysis_saved", has_analysis=bool(analysis))
        store.save_context(
            RollingContext(
                max_recent_keep=max(
                    40,
                    self._runtime.config.pipeline.rolling_context_segments,
                )
            ).to_dict()
        )

        # The manifest marks successful initialization and must be committed atomically last.
        manifest["initialized"] = True
        store.save_manifest(manifest)
        self._runtime.bind_timing(store)
        store.finish_initialization()
        store.log_event(
            "run_initialized",
            input_path=input_path,
            run_dir=store.run_dir,
            title=doc.title,
            fmt=doc.fmt,
            source_lang=doc.source_lang,
            target_lang=doc.target_lang,
            chapters=len(doc.chapters),
            config={
                "review": self._runtime.config.pipeline.review,
                "polish": self._runtime.config.pipeline.polish,
                "book_understanding": self._runtime.config.pipeline.book_understanding,
                "review_concurrency": self._runtime.config.pipeline.review_concurrency,
                "review_output_retries": (self._runtime.config.pipeline.review_output_retries),
            },
        )
        return store

    def activate(self, store: Storage, *, phase: Phase | None = None) -> dict[str, Any]:
        """Restore manifest languages, propagate them to all agents and return the manifest."""
        store.recover_usage()
        manifest = store.load_manifest()
        self._runtime.apply_manifest_languages(manifest)
        if phase == "translation" and translation_revision(store, self._runtime.config):
            self._rebuild_analysis(store, manifest)
        return manifest

    def _rebuild_analysis(self, store: Storage, manifest: dict[str, Any]) -> None:
        """Refresh built-in guidance while preserving formal targets and glossary."""
        chapters = [store.load_chapter(row["index"]) for row in manifest["chapters"]]
        if self._runtime.config.pipeline.book_understanding:
            # Reuse the single prescan path so glossary patching and policy stamps apply here too.
            self.ensure_understanding(store)
        # Assemble only the source samples; formal chapters are never rewritten here.
        document = Document(
            title=manifest.get("title", ""),
            source_lang=self._runtime.config.source_lang,
            target_lang=self._runtime.config.target_lang,
            fmt=manifest.get("fmt", "text"),
            chapters=chapters,
        )
        style_policy = self._runtime.config.language_policy("analysis").task_fingerprint("analyzer")
        analysis = store.load_analysis() or {}
        if analysis.get("style_policy") != style_policy:
            analysis = self._runtime.analyzer.analyze(self.sample_text(document))
        analysis["style_policy"] = style_policy
        analysis["language_policy"] = (
            f"language-policies/{self._runtime.config.language_policy('analysis').fingerprint}.json"
        )
        store.save_analysis(analysis)
        commit_revision(store, self._runtime.config)
        store.log_event(
            "language_policy_revision_refreshed", rebuild="analysis", completed_targets="preserved"
        )

    def detect_language_ai(self, doc) -> str:
        """Detect the primary source language with the model; return its code or empty on
        failure.
        """
        # Use unlabeled source samples so sampling labels cannot contaminate language detection.
        sample = self.sample_text(doc, labeled=False)[:1500]
        if not sample.strip():
            return ""
        system = render("language_detector_system")
        try:
            data = self._runtime.client.complete_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": sample},
                ],
                operation="language.detect",
            )
            code = (data.get("language") if isinstance(data, dict) else "") or ""
            return normalize_language(str(code))
        except Exception:  # noqa: BLE001 - provider errors mean detection failed
            return ""

    @staticmethod
    def sample_text(doc, *, labeled: bool = True) -> str:
        """Select style samples from the beginning, middle and end with labels when requested.
        For language detection, return one pure source sample without labels so the label
        language cannot bias detection.
        """
        texts = ["\n".join(s.source for s in ch.text_segments) for ch in doc.chapters]
        texts = [t for t in texts if len(t) > 200]
        if not texts:  # Fallback when every chapter is short.
            joined = "\n".join(s.source for ch in doc.chapters[:2] for s in ch.text_segments)
            return joined[:6000]
        if not labeled:
            return texts[0][:6000]
        picks = [
            (0, "Opening sample"),
            (len(texts) // 2, "Middle sample"),
            (len(texts) - 1, "Ending sample"),
        ]
        parts: list[str] = []
        seen: set[int] = set()
        for idx, tag in picks:
            if idx in seen:  # Deduplicate samples for short books with one or two chapters.
                continue
            seen.add(idx)
            t = texts[idx]
            chunk = t[-2800:] if tag == "Ending sample" else t[:2800]
            parts.append(f"【{tag}】\n{chunk}")
        return "\n\n".join(parts)

    # Book-understanding prescan: chapter digests and a whole-book synopsis.
    # Version 3 embeds glossary-aware prompts; glossary_fp maps source→target at generation
    # time so that modifying an existing translation patches the cached digest in place.
    SOURCE_DIGEST_V = 3
    BOOK_SYNOPSIS_V = 3

    @staticmethod
    def _glossary_fp(terms: list) -> dict[str, str]:
        """Map source→target for glossary fingerprinting."""
        return {
            t.source: t.target
            for t in terms
            if getattr(t, "source", "") and getattr(t, "target", "")
        }

    @staticmethod
    def _glossary_edits(
        stored: dict[str, str], glossary_fp: dict[str, str]
    ) -> list[tuple[str, str, str]]:
        """Return (source, old_target, new_target) for terms whose target changed."""
        edits: list[tuple[str, str, str]] = []
        for src, old_tgt in stored.items():
            new_tgt = glossary_fp.get(src)
            if new_tgt is not None and new_tgt != old_tgt:
                edits.append((src, old_tgt, new_tgt))
        return edits

    @staticmethod
    def _patch_text(text: str, edits: list[tuple[str, str, str]]) -> tuple[str, dict[str, str]]:
        """Patch glossary renderings by source anchor; never blind-replace targets.

        Digests are target-language prose. Identity is the source form. Only rewrite a
        target when the source form appears next to it (``target（source）`` style).
        Returns the patched text and the source→new_target map that was applied.
        Unapplied sources stay out of the map so the fingerprint still marks the digest
        stale and forces regeneration instead of a wrong global replace.
        """
        applied: dict[str, str] = {}
        for source, old_tgt, new_tgt in edits:
            if not source or not old_tgt or not new_tgt:
                continue
            if source not in text:
                continue
            replacements = (
                (f"{old_tgt}（{source}）", f"{new_tgt}（{source}）"),
                (f"{old_tgt} ({source})", f"{new_tgt} ({source})"),
                (f"{old_tgt}({source})", f"{new_tgt}({source})"),
                (f"{old_tgt}（{source}）", f"{new_tgt}（{source}）"),
                (f"{old_tgt} ({source})", f"{new_tgt} ({source})"),
                (f"{old_tgt}「{source}」", f"{new_tgt}「{source}」"),
                (f"{old_tgt} / {source}", f"{new_tgt} / {source}"),
                (f"{source} / {old_tgt}", f"{source} / {new_tgt}"),
            )
            patched = False
            for old, new in replacements:
                if old in text:
                    text = text.replace(old, new)
                    patched = True
            if patched:
                applied[source] = new_tgt
        return text, applied

    @staticmethod
    def _merge_glossary_fp(
        stored: dict[str, str],
        glossary_fp: dict[str, str],
        applied: dict[str, str],
    ) -> dict[str, str]:
        """Keep unapplied sources stale so a later pass regenerates instead of guessing."""
        merged = dict(stored)
        merged.update(applied)
        return merged

    @staticmethod
    def _digest_is_current(meta: dict, glossary_fp: dict[str, str], policy: str) -> bool:
        digest = meta.get("source_digest") or ""
        version = meta.get("source_digest_v", 1)
        if not (str(digest).strip() and isinstance(version, int) and version >= 3):
            return False
        stored = meta.get("source_digest_gf") or {}
        if not isinstance(stored, dict):
            return False
        # Only an existing pair's target change invalidates; new terms are fine.
        for src, tgt in stored.items():
            if glossary_fp.get(src) != tgt:
                return False
        # A digest without a policy stamp predates policy tracking. It already carries the
        # glossary snapshot validated above, and this upgrade leaves the digest prompts
        # untouched, so it stays reusable and receives its stamp on the next write.
        recorded = meta.get("source_digest_policy")
        return not recorded or recorded == policy

    @staticmethod
    def _synopsis_is_current(analysis: dict, glossary_fp: dict[str, str], inputs: str) -> bool:
        synopsis = analysis.get("book_synopsis") or ""
        version = analysis.get("book_synopsis_v", 1)
        if not (str(synopsis).strip() and isinstance(version, int) and version >= 3):
            return False
        stored = analysis.get("book_synopsis_gf") or {}
        if not isinstance(stored, dict):
            return False
        for src, tgt in stored.items():
            if glossary_fp.get(src) != tgt:
                return False
        # The synopsis also depends on the style brief, the digests and the prompts that
        # produced it, so any of those changing makes it stale.
        recorded = analysis.get("book_synopsis_inputs")
        return not recorded or recorded == inputs

    def ensure_understanding(
        self,
        store: Storage,
        progress: ProgressFn | None = None,
    ) -> str:
        """Prescan source chapters into chapter.meta digests and an analysis synopsis.
        Skip existing results for idempotent resume. Return the synopsis for translation
        prompts, or empty when book_understanding is disabled.
        """
        if not self._runtime.config.pipeline.book_understanding:
            store.log_event("book_understanding_skipped", reason="disabled")
            return ""
        manifest = store.load_manifest()
        chapters = manifest.get("chapters", [])

        # Snapshot glossary terms once; the fingerprint gates digest reuse across edits.
        glossary_terms = list(store.all_terms()) if hasattr(store, "all_terms") else []
        glossary_fp = self._glossary_fp(glossary_terms)

        loaded = {
            c.get("index", i): store.load_chapter(c.get("index", i)) for i, c in enumerate(chapters)
        }
        # Digest and synopsis reuse need both identities: the glossary snapshot that steered the
        # text, and the prompt/rules revision that produced it.
        analysis_plan = self._runtime.config.language_policy("analysis")
        digest_policy = analysis_plan.task_fingerprint("chapter_digest")
        synopsis_policy = analysis_plan.task_fingerprint("book_synopsis")

        # Phase 1: patch existing digests when glossary targets changed (no LLM call).
        patched_any = False
        for ci, ch in loaded.items():
            meta = ch.meta
            digest = meta.get("source_digest") or ""
            version = meta.get("source_digest_v", 1)
            stored = meta.get("source_digest_gf") or {}
            if not (str(digest).strip() and isinstance(version, int) and version >= 3):
                continue
            if not isinstance(stored, dict) or not stored:
                continue
            edits = self._glossary_edits(stored, glossary_fp)
            if not edits:
                continue
            patched_digest, applied = self._patch_text(str(digest), edits)
            if not applied:
                continue
            meta["source_digest"] = patched_digest
            meta["source_digest_gf"] = self._merge_glossary_fp(stored, glossary_fp, applied)
            store.save_chapter(ch)
            patched_any = True
            store.log_event(
                "book_understanding_digest_patched",
                chapter=ci,
                edits=[{"source": s, "old": o, "new": n} for s, o, n in edits if s in applied],
                skipped=[s for s, _o, _n in edits if s not in applied],
            )
        if patched_any:
            # Reload patched digests for the synopsis step.
            loaded = {
                c.get("index", i): store.load_chapter(c.get("index", i))
                for i, c in enumerate(chapters)
            }

        # Phase 2: generate missing or outdated digests.
        todo = [
            (ci, "\n".join(s.source for s in ch.text_segments))
            for ci, ch in loaded.items()
            if not self._digest_is_current(ch.meta, glossary_fp, digest_policy)
        ]
        if todo:
            store.log_event(
                "book_understanding_chapter_digest_started",
                chapters=[ci for ci, _ in todo],
                workers=max(1, self._runtime.config.pipeline.prescan_concurrency),
            )
            workers = max(1, self._runtime.config.pipeline.prescan_concurrency)
            if progress:
                progress(0, len(todo), "Prescanning chapter digests")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(self._runtime.synopsizer.digest_chapter, src, glossary_terms): ci
                    for ci, src in todo
                }
                for n_done, fut in enumerate(as_completed(futs), 1):
                    ci = futs[fut]
                    digest = fut.result()  # _ask_text already returns an empty fallback on failure.
                    if not str(digest).strip():
                        # Never cache a failure: keep the chapter out of date so a later
                        # run retries it instead of reusing an empty digest.
                        store.log_event(
                            "book_understanding_chapter_digest_failed",
                            chapter=ci,
                        )
                        if progress:
                            progress(n_done, len(todo), "Prescanning chapter digests")
                        continue
                    loaded[ci].meta["source_digest"] = digest
                    loaded[ci].meta["source_digest_v"] = self.SOURCE_DIGEST_V
                    loaded[ci].meta["source_digest_gf"] = glossary_fp
                    loaded[ci].meta["source_digest_policy"] = digest_policy
                    store.save_chapter(loaded[ci])
                    store.log_event(
                        "book_understanding_chapter_digest_saved",
                        chapter=ci,
                        digest=loaded[ci].meta["source_digest"],
                    )
                    if progress:
                        progress(n_done, len(todo), "Prescanning chapter digests")

        # Assemble in manifest chapter order, independent of worker completion order.
        digests = [
            loaded[c.get("index", i)].meta.get("source_digest", "") or ""
            for i, c in enumerate(chapters)
        ]

        # Every translatable chapter needs a digest before translation: downstream prompts
        # and the synopsis depend on them, so a partial prescan must fail loudly rather
        # than translate with missing context.
        missing_digests = [
            ci
            for ci, ch in loaded.items()
            if any((seg.source or "").strip() for seg in ch.text_segments)
            and not str(ch.meta.get("source_digest", "") or "").strip()
        ]
        if missing_digests:
            store.log_event(
                "book_understanding_incomplete",
                chapters=sorted(missing_digests),
            )
            raise ValueError(
                "Chapter digests could not be generated for chapters: "
                + ", ".join(str(ci) for ci in sorted(missing_digests))
            )

        analysis = store.load_analysis() or {}
        synopsis = str(analysis.get("book_synopsis", "") or "")

        # Phase 3: patch existing synopsis when glossary targets changed (source-anchored).
        syn_stored = analysis.get("book_synopsis_gf") or {}
        syn_version = analysis.get("book_synopsis_v", 1)
        if (
            synopsis.strip()
            and isinstance(syn_version, int)
            and syn_version >= 3
            and isinstance(syn_stored, dict)
            and syn_stored
        ):
            syn_edits = self._glossary_edits(syn_stored, glossary_fp)
            if syn_edits:
                patched_syn, syn_applied = self._patch_text(synopsis, syn_edits)
                if syn_applied:
                    synopsis = patched_syn
                    analysis["book_synopsis"] = synopsis
                    analysis["book_synopsis_gf"] = self._merge_glossary_fp(
                        syn_stored, glossary_fp, syn_applied
                    )
                    store.save_analysis(analysis)
                    store.log_event(
                        "book_synopsis_patched",
                        edits=[
                            {"source": s, "old": o, "new": n}
                            for s, o, n in syn_edits
                            if s in syn_applied
                        ],
                        skipped=[s for s, _o, _n in syn_edits if s not in syn_applied],
                    )

        # Phase 4: generate synopsis when missing, incomplete or outdated.
        # The synopsis tracks the prompts and the style brief that shaped it. Glossary edits are
        # handled by the fingerprint check plus in-place patching above, and source changes
        # invalidate the whole run, so neither belongs in this hash.
        style = self._runtime.analyzer.style_brief(analysis)
        synopsis_inputs = content_hash({"policy": synopsis_policy, "style": style})
        current = self._synopsis_is_current(analysis, glossary_fp, synopsis_inputs)
        if (not current or not _synopsis_complete(synopsis)) and any(d.strip() for d in digests):
            if progress:
                progress(0, 0, "Generating whole-book synopsis…")
            generated = self._runtime.synopsizer.book_synopsis(digests, style, glossary_terms)
            if _synopsis_complete(generated):
                synopsis = generated
                analysis["book_synopsis"] = synopsis
                analysis["book_synopsis_v"] = self.BOOK_SYNOPSIS_V
                analysis["book_synopsis_gf"] = glossary_fp
                analysis["book_synopsis_inputs"] = synopsis_inputs
                store.save_analysis(analysis)
                store.log_event("book_synopsis_saved", synopsis=synopsis)
            else:
                # A failed or truncated regeneration must not overwrite a working synopsis, and
                # an outdated one must not reach translation prompts either.
                store.log_event("book_synopsis_failed", incomplete=True)
                if not current:
                    synopsis = ""
                if progress:
                    progress(0, 0, "Whole-book synopsis unavailable")
        return str(synopsis or "")
