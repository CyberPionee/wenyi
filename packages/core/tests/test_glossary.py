"""Glossary tests."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from wenyi_core.agents.prompts import render_glossary, strip_empty_sections
from wenyi_core.glossary.resolver import keep_current_terms
from wenyi_core.glossary.store import (
    MANUAL_STATUS,
    TYPE_APPELLATION,
    TYPE_PERSON,
    UPSERT_CONFLICT,
    UPSERT_DROP,
    UPSERT_FILL,
    UPSERT_INSERT,
    UPSERT_MANUAL_WRITE,
    UPSERT_MERGE,
    GlossaryStore,
    GlossaryTerm,
    classify_upsert,
    merge_always_on,
    source_matches_text,
    upsert_result,
)


class TestGlossary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = GlossaryStore(os.path.join(self.tmp.name, "g.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_insert_and_lookup(self):
        r = self.store.upsert_term(
            GlossaryTerm(
                source="綾小路",
                target="绫小路",
                type=TYPE_PERSON,
                gender="male",
                aliases=["綾小路くん"],
                reading="あやのこうじ",
            ),
            chapter=0,
        )
        self.assertEqual(r, "inserted")
        t = self.store.get_term("綾小路")
        assert t is not None
        self.assertEqual(t.target, "绫小路")
        self.assertEqual(t.gender, "male")

    def test_terms_in_matches_alias(self):
        self.store.upsert_term(
            GlossaryTerm(source="綾小路", target="绫小路", aliases=["綾小路くん"])
        )
        hits = self.store.terms_in(
            self.store.all_terms(), "「おはよう、綾小路くん」と堀北が言った。"
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "綾小路")

    def test_terms_in_normalizes_case_and_character_width(self):
        self.store.upsert_term(GlossaryTerm(source="OpenAI", target="开放人工智能"))
        self.store.upsert_term(GlossaryTerm(source="ＡＢＣ", target="ABC 组织"))

        hits = self.store.terms_in(self.store.all_terms(), "openai 与 ABC")

        self.assertEqual(
            {term.source for term in hits},
            {"OpenAI", "ＡＢＣ"},
        )

    def test_ascii_source_match_respects_word_boundaries(self):
        self.assertTrue(source_matches_text("Ann", "Ann opened the door."))
        self.assertTrue(source_matches_text("ANN", "ann opened the door."))
        self.assertFalse(source_matches_text("Ann", "Anna opened the door."))

    def test_cyrillic_source_match_respects_word_boundaries(self):
        self.assertTrue(source_matches_text("гад", "Этот гад снова пришёл."))
        self.assertFalse(source_matches_text("гад", "Этот гадкий человек снова пришёл."))

    def test_appellation_does_not_match_bare_name_alias(self):
        self.store.upsert_term(
            GlossaryTerm(
                source="夏帆ちゃん",
                target="小夏帆",
                type=TYPE_APPELLATION,
                aliases=["夏帆"],
            )
        )
        self.assertEqual(self.store.terms_in(self.store.all_terms(), "夏帆は窓の外を見た。"), [])
        hits = self.store.terms_in(self.store.all_terms(), "「夏帆ちゃん」と母親が言った。")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "夏帆ちゃん")

    def test_recurring_terms_require_two_full_text_occurrences(self):
        terms = [
            GlossaryTerm(source="唯一术语", target="Unique"),
            GlossaryTerm(source="重复术语", target="Repeated"),
            GlossaryTerm(
                source="AliasCanonical",
                target="Alias",
                aliases=["别名"],
            ),
            GlossaryTerm(
                source="夏帆ちゃん",
                target="小夏帆",
                type=TYPE_APPELLATION,
                aliases=["夏帆"],
            ),
        ]
        corpus = "唯一术语。重复术语再次成为重复术语。别名先来，别名再来。夏帆出现两次，夏帆。"

        recurring = GlossaryStore.recurring_terms(terms, corpus)

        self.assertEqual(
            {term.source for term in recurring},
            {"重复术语", "AliasCanonical"},
        )

        overlapping_alias = GlossaryTerm(
            source="夏帆",
            target="Kaho",
            aliases=["夏帆ちゃん"],
        )
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [overlapping_alias],
                "夏帆ちゃん只在全文出现一次。",
            ),
            [],
        )

        cyrillic = GlossaryTerm(source="гад", target="畜生")
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [cyrillic],
                "Один гад ушёл, но гадкий человек остался гадким.",
            ),
            [],
        )
        self.assertEqual(
            GlossaryStore.recurring_terms(
                [cyrillic],
                "Один гад ушёл, затем другой гад пришёл.",
            ),
            [cyrillic],
        )

    def test_upsert_fills_empty_target_without_conflict(self):
        self.store.upsert_term(GlossaryTerm(source="Ann", target=""), chapter=0)
        result = self.store.upsert_term(GlossaryTerm(source="Ann", target="安"), chapter=1)
        self.assertEqual(result, "updated")
        term = self.store.get_term("Ann")
        assert term is not None
        self.assertEqual(term.target, "安")
        self.assertEqual(term.status, "ok")
        self.assertEqual(len(self.store.open_conflicts()), 0)
        # A different non-empty target remains a conflict and does not overwrite.
        result = self.store.upsert_term(GlossaryTerm(source="Ann", target="安娜"), chapter=2)
        self.assertEqual(result, "conflict")
        self.assertEqual(self.store.get_term("Ann").target, "安")

    def test_conflict_keeps_current_until_resolved(self):
        self.store.upsert_term(GlossaryTerm(source="堀北", target="堀北"), chapter=0)
        # An alternate translation preserves the established mapping and records a candidate.
        r = self.store.upsert_term(GlossaryTerm(source="堀北", target="掘北"), chapter=1)
        self.assertEqual(r, "conflict")
        term = self.store.get_term("堀北")
        assert term is not None
        self.assertEqual(term.target, "堀北")
        self.assertEqual(len(self.store.open_conflicts()), 1)

        self.assertTrue(self.store.resolve_term("堀北", "掘北"))
        self.store.mark_conflicts_resolved("堀北")
        term = self.store.get_term("堀北")
        assert term is not None
        self.assertEqual(term.target, "掘北")
        # Resolving is the operator's decision, so it locks the term like an explicit edit.
        self.assertEqual(term.status, MANUAL_STATUS)
        self.assertEqual(self.store.open_conflicts(), [])
        # A locked term stops the extraction pass from re-opening what the operator closed.
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="堀北", target="堀北"), chapter=2),
            "unchanged",
        )
        self.assertEqual(self.store.open_conflicts(), [])
        self.assertEqual(self.store.get_term("堀北").target, "掘北")

    def test_repeated_proposal_records_one_conflict(self):
        """A later batch re-proposing the same alternative must not add another row."""
        self.store.upsert_term(GlossaryTerm(source="アメ", target="美国"), chapter=29)
        for chapter in range(29, 42):
            self.assertEqual(
                self.store.upsert_term(GlossaryTerm(source="アメ", target="阿梅"), chapter=chapter),
                "conflict",
            )
        conflicts = self.store.open_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["proposed_target"], "阿梅")
        # A genuinely different alternative is a new decision and is still recorded.
        self.store.upsert_term(GlossaryTerm(source="アメ", target="美利坚"), chapter=42)
        self.assertEqual(len(self.store.open_conflicts()), 2)

    def test_classify_upsert_covers_every_branch(self):
        """Every storage backend routes through this decision, so each branch is pinned."""
        plain = GlossaryTerm(source="s", target="编甲")
        locked = GlossaryTerm(source="s", target="编甲", status=MANUAL_STATUS)
        self.assertEqual(classify_upsert(None, plain), UPSERT_INSERT)
        self.assertEqual(
            classify_upsert(plain, GlossaryTerm(source="s", target="编乙", status=MANUAL_STATUS)),
            UPSERT_MANUAL_WRITE,
        )
        self.assertEqual(
            classify_upsert(locked, GlossaryTerm(source="s", target="编乙")), UPSERT_DROP
        )
        self.assertEqual(classify_upsert(GlossaryTerm(source="s", target=""), plain), UPSERT_FILL)
        self.assertEqual(
            classify_upsert(plain, GlossaryTerm(source="s", target="编甲")), UPSERT_MERGE
        )
        self.assertEqual(
            classify_upsert(plain, GlossaryTerm(source="s", target="编乙")), UPSERT_CONFLICT
        )
        self.assertEqual(upsert_result(UPSERT_INSERT), "inserted")
        self.assertEqual(upsert_result(UPSERT_MANUAL_WRITE), "updated")
        self.assertEqual(upsert_result(UPSERT_DROP), "unchanged")
        self.assertEqual(upsert_result(UPSERT_FILL), "updated")
        self.assertEqual(upsert_result(UPSERT_MERGE), "unchanged")
        self.assertEqual(upsert_result(UPSERT_CONFLICT), "conflict")

    def test_locked_term_makes_its_recorded_conflicts_moot(self):
        """Rows recorded before the lock existed must not gate a run."""
        self.store.upsert_term(GlossaryTerm(source="海豚", target="海豚酒店"), chapter=0)
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="海豚", target="海豚旅店"), chapter=1),
            "conflict",
        )
        self.assertEqual(len(self.store.open_conflicts()), 1)

        # A lock applied outside upsert leaves the row unresolved, as legacy state does.
        self.store.conn.execute(
            "UPDATE glossary SET status=? WHERE source=?", (MANUAL_STATUS, "海豚")
        )
        self.store.conn.commit()
        self.assertEqual(self.store.open_conflicts(), [])
        self.assertEqual(self.store.stats()["open_conflicts"], 0)

    def test_keep_current_terms_settles_every_open_conflict(self):
        """Bulk settlement keeps each established target and locks the terms it settles."""
        self.store.upsert_term(GlossaryTerm(source="甲", target="Jia"), chapter=0)
        self.store.upsert_term(GlossaryTerm(source="乙", target="Yi"), chapter=0)
        for source in ("甲", "乙"):
            self.assertEqual(
                self.store.upsert_term(GlossaryTerm(source=source, target="别的"), chapter=1),
                "conflict",
            )
        self.store.upsert_term(GlossaryTerm(source="丁", target="Ding"), chapter=0)
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="丁", target="别的"), chapter=1), "conflict"
        )
        # A conflict whose term is gone must stay visible instead of being discarded.
        self.store.conn.execute("DELETE FROM glossary WHERE source=?", ("丁",))
        self.store.conn.commit()

        settled = keep_current_terms(self.store)
        self.assertEqual(sorted(record["source"] for record in settled), ["乙", "甲"])
        # Each record carries the rejected proposal so the caller can rewrite the translation.
        self.assertEqual(
            {record["source"]: (record["target"], record["rejected"]) for record in settled},
            {"甲": ("Jia", ["别的"]), "乙": ("Yi", ["别的"])},
        )
        first = self.store.get_term("甲")
        assert first is not None
        self.assertEqual(first.target, "Jia")
        # Settling is the operator's decision, so the term is locked against later proposals.
        self.assertEqual(first.status, MANUAL_STATUS)
        self.assertEqual(self.store.get_term("乙").target, "Yi")
        self.assertEqual([row["source"] for row in self.store.open_conflicts()], ["丁"])
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="甲", target="别的"), chapter=2), "unchanged"
        )

    def test_manual_target_outranks_every_later_proposal(self):
        """A target an operator set is final: proposals are dropped, not recorded."""
        self.store.upsert_term(
            GlossaryTerm(source="海豚", target="海豚旅店", status=MANUAL_STATUS), chapter=0
        )
        # The extraction pass keeps proposing its own reading of the same source.
        for _ in range(3):
            self.assertEqual(
                self.store.upsert_term(GlossaryTerm(source="海豚", target="海豚酒店"), chapter=1),
                "unchanged",
            )
        term = self.store.get_term("海豚")
        assert term is not None
        self.assertEqual(term.target, "海豚旅店")
        self.assertEqual(term.status, MANUAL_STATUS)
        # No review backlog accumulates against a decision already made.
        self.assertEqual(self.store.open_conflicts(), [])

    def test_manual_edit_clears_the_recorded_conflict(self):
        """Editing a term by hand retires the conflict it was carrying."""
        self.store.upsert_term(GlossaryTerm(source="海豚", target="海豚酒店"), chapter=0)
        self.assertEqual(
            self.store.upsert_term(GlossaryTerm(source="海豚", target="海豚旅店"), chapter=1),
            "conflict",
        )
        self.assertEqual(len(self.store.open_conflicts()), 1)

        self.store.upsert_term(
            GlossaryTerm(source="海豚", target="海豚旅店", status=MANUAL_STATUS), chapter=1
        )
        term = self.store.get_term("海豚")
        assert term is not None
        self.assertEqual(term.target, "海豚旅店")
        self.assertEqual(term.status, MANUAL_STATUS)
        self.assertEqual(self.store.open_conflicts(), [])

    def test_concurrent_upserts_make_one_atomic_conflict_decision(self):
        path = os.path.join(self.tmp.name, "concurrent.db")
        initial = GlossaryStore(path)
        initial.close()
        barrier = threading.Barrier(2)

        def write(target: str) -> str:
            store = GlossaryStore(path)
            try:
                barrier.wait()
                return store.upsert_term(GlossaryTerm(source="Name", target=target), chapter=1)
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(write, ["译名甲", "译名乙"]))

        check = GlossaryStore(path)
        try:
            self.assertCountEqual(results, ["inserted", "conflict"])
            self.assertEqual(len(check.all_terms()), 1)
            self.assertEqual(len(check.open_conflicts()), 1)
        finally:
            check.close()

    def test_stats(self):
        self.store.upsert_term(GlossaryTerm(source="A", target="甲"))
        s = self.store.stats()
        self.assertEqual(s, {"terms": 1, "open_conflicts": 0})

    def test_all_terms_preserves_insert_order_not_type_source_sort(self):
        """Insertion order controls all_terms so new entries cannot disrupt prompt prefix
        caching.
        """
        # Insert terms in an order that differs from type/source sorting deliberately.
        self.store.upsert_term(
            GlossaryTerm(source="乙", target="Yi", type="term"),
            chapter=0,
        )
        self.store.upsert_term(
            GlossaryTerm(source="甲", target="Jia", type=TYPE_PERSON),
            chapter=0,
        )
        self.assertEqual(
            [term.source for term in self.store.all_terms()],
            ["乙", "甲"],
        )
        # Human resolution or same-target field merging must not change insertion position.
        self.assertTrue(self.store.resolve_term("乙", "Yi-updated"))
        terms = self.store.all_terms()
        self.assertEqual([term.source for term in terms], ["乙", "甲"])
        self.assertEqual(terms[0].target, "Yi-updated")
        # New terms may only append to the end.
        self.store.upsert_term(
            GlossaryTerm(source="丙", target="Bing", type=TYPE_PERSON),
            chapter=1,
        )
        self.assertEqual(
            [term.source for term in self.store.all_terms()],
            ["乙", "甲", "丙"],
        )

    def test_render_glossary_includes_truncated_note(self):
        terms = [
            GlossaryTerm(source="Ann", target="安", type=TYPE_PERSON, note="Main heroine"),
            GlossaryTerm(source="Bob", target="鲍勃", type=TYPE_PERSON, note=""),
            GlossaryTerm(source="Cy", target="赛", type=TYPE_PERSON, note="x" * 50),
        ]
        rendered = render_glossary(terms, max_note_chars=12)
        self.assertIn("Note: Main heroine", rendered)
        self.assertNotIn("Note:", rendered.split("- Bob")[1].split("\n")[0])
        self.assertIn(f"Note: {'x' * 12}", rendered)
        self.assertNotIn("x" * 13, rendered)
        without_notes = render_glossary(terms, include_note=False)
        self.assertNotIn("Note:", without_notes)

    def test_merge_always_on_appends_frequent_locked_persons(self):
        chapter_hit = GlossaryTerm(source="Local", target="本地", type=TYPE_PERSON)
        hero = GlossaryTerm(source="Ann", target="安", type=TYPE_PERSON, note="hero")
        rare = GlossaryTerm(source="OneOff", target="路人", type=TYPE_PERSON)
        place = GlossaryTerm(source="Tokyo", target="东京", type="term")
        conflicted = GlossaryTerm(source="Cy", target="赛", type=TYPE_PERSON, status="conflict")
        all_terms = [chapter_hit, hero, rare, place, conflicted]
        corpus = "Ann meets Ann and Ann again. Local stays. Cy appears Cy."

        merged = merge_always_on([chapter_hit], all_terms, corpus, min_occurrences=3)
        self.assertEqual([term.source for term in merged], ["Local", "Ann"])

    def test_merge_always_on_respects_type_and_max_cap(self):
        persons = [
            GlossaryTerm(source=f"P{i}", target=f"人{i}", type=TYPE_PERSON) for i in range(3)
        ]
        all_terms = [*persons, GlossaryTerm(source="Org", target="组织", type="term")]
        corpus = " ".join(f"P{i} P{i} P{i}" for i in range(3)) + " Org Org Org"

        merged = merge_always_on(
            [],
            all_terms,
            corpus,
            always_types=[TYPE_PERSON],
            min_occurrences=3,
            max_always=2,
        )
        self.assertEqual([term.source for term in merged], ["P0", "P1"])

    def test_strip_empty_sections_removes_blank_headings(self):
        text = (
            "## Plot\nHolden leaves school.\n\n"
            "## Characters\n   \n\n"
            "## Foreshadowing\n\n"
            "## Address\nHe calls Phoebe.\n"
        )
        stripped = strip_empty_sections(text)
        self.assertIn("## Plot", stripped)
        self.assertIn("## Address", stripped)
        self.assertNotIn("## Characters", stripped)
        self.assertNotIn("## Foreshadowing", stripped)

    def test_strip_empty_sections_keeps_nonempty_sections(self):
        text = "## Plot\nSomething happens.\n## Characters\nAnn appears.\n"
        self.assertEqual(strip_empty_sections(text).count("## "), 2)

    def test_strip_empty_sections_handles_no_headings(self):
        self.assertEqual(strip_empty_sections("Just a sentence."), "Just a sentence.")

    def test_strip_empty_sections_handles_empty_input(self):
        self.assertEqual(strip_empty_sections(""), "")
        self.assertEqual(strip_empty_sections("   \n  "), "")


if __name__ == "__main__":
    unittest.main()
