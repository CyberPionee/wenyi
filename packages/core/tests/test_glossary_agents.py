"""Offline analyzer, glossary-extraction and rolling-context tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from wenyi_core.agents.analyzer import Analyzer
from wenyi_core.agents.base import Agent
from wenyi_core.config import Config
from wenyi_core.glossary.extractor import (
    GlossaryExtractor,
    TranslatedSegmentEvidence,
)
from wenyi_core.glossary.store import GlossaryStore, GlossaryTerm
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.retrying import TruncatedResponseError
from wenyi_core.pipeline.context import RollingContext


def _cfg():
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "preset": "fake",
                "models": {
                    "default_strong": {"provider": "default", "model": "p"},
                    "default_cheap": {"provider": "default", "model": "f"},
                },
            },
        }
    )


class TestAgentCollectionNormalization(unittest.TestCase):
    def test_invalid_collection_logs_context_without_response_content(self):
        for value in (0, 1.5, True, False, "private model text", {"private model text": 1}):
            with self.subTest(value=value):
                with self.assertLogs("wenyi_core.agents.base", level="WARNING") as logs:
                    result = Agent.dict_items(value, operation="glossary.extract", field="terms")

                self.assertEqual(result, [])
                self.assertEqual(len(logs.records), 1)
                message = logs.records[0].getMessage()
                self.assertIn("operation=glossary.extract", message)
                self.assertIn("field=terms", message)
                self.assertIn(f"got {type(value).__name__}", message)
                self.assertNotIn("private model text", message)

    def test_missing_and_valid_collections_do_not_log(self):
        with self.assertNoLogs("wenyi_core.agents.base", level="WARNING"):
            for value in (None, [], [{"source": "a", "target": "b"}]):
                self.assertEqual(Agent.dict_items(value), value or [])

    def test_filtered_members_log_one_count_without_response_content(self):
        valid = {"source": "private source", "target": "private translation"}
        with self.assertLogs("wenyi_core.agents.base", level="WARNING") as logs:
            result = Agent.dict_items(
                ["private model text", 1, valid], operation="glossary.extract", field="terms"
            )

        self.assertEqual(result, [valid])
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn("operation=glossary.extract", message)
        self.assertIn("field=terms", message)
        self.assertIn("2 non-object items", message)
        self.assertNotIn("private", message)

    def test_non_list_model_collections_are_rejected(self):
        for value in (None, 0, 1, 0.0, 1.5, "invalid", {}):
            with self.subTest(value=value):
                self.assertEqual(Agent.dict_items(value), [])

    def test_json_arrays_keep_only_dictionary_members(self):
        valid = {"source": "a", "target": "b"}

        self.assertEqual(Agent.dict_items([]), [])
        self.assertEqual(Agent.dict_items([valid]), [valid])
        self.assertEqual(
            Agent.dict_items([None, 1, 1.5, "invalid", {}, valid]),
            [{}, valid],
        )


class TestAnalyzer(unittest.TestCase):
    def test_style_brief_omits_snapshot_target_and_preserves_character_guidance(self):
        analyzer = Analyzer(FakeClient(), _cfg())
        brief = analyzer.style_brief(
            {
                "characters": [
                    {
                        "source": "H.H.",
                        "target": "Stale snapshot name",
                        "gender": " male ",
                        "note": "Speaks in restrained, formal sentences.",
                    }
                ]
            }
        )

        self.assertNotIn("Stale snapshot name", brief)
        self.assertIn("H.H.", brief)
        self.assertIn("male", brief)
        self.assertIn("Speaks in restrained, formal sentences.", brief)

    def test_analyze_and_seed(self):
        analysis = {
            "genre": "校园",
            "tone": "冷峻第三人称",
            "style_guide": "保持克制",
            "characters": [
                {
                    "source": "綾小路",
                    "target": "绫小路",
                    "gender": "male",
                    "reading": "あやのこうじ",
                    "note": "第一人称用俺",
                }
            ],
            "terms": [{"source": "高度育成高校", "target": "高度育成高中", "type": "organization"}],
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(analysis, ensure_ascii=False))
        a = Analyzer(client, _cfg())
        result = a.analyze("……样章……")
        self.assertEqual(result["genre"], "校园")

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            n = a.seed_glossary(store, result)
            self.assertEqual(n, 2)
            character = store.get_term("綾小路")
            organization = store.get_term("高度育成高校")
            self.assertIsNotNone(character)
            self.assertIsNotNone(organization)
            assert character is not None
            assert organization is not None
            self.assertEqual(character.gender, "male")
            self.assertEqual(organization.type, "organization")
            store.close()

        brief = a.style_brief(result)
        self.assertIn("綾小路", brief)
        self.assertNotIn("绫小路", brief)

    def test_malformed_collection_items_are_filtered(self):
        analysis = {
            "genre": {"unexpected": True},
            "characters": ["bad", {"source": "綾小路", "target": "绫小路"}],
            "terms": [1, {"source": "学校", "target": "学校", "type": {"bad": 1}}],
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(analysis, ensure_ascii=False))
        analyzer = Analyzer(client, _cfg())
        result = analyzer.analyze("……样章……")

        self.assertEqual(result["genre"], "")
        self.assertEqual(len(result["characters"]), 1)
        self.assertEqual(len(result["terms"]), 1)
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            self.assertEqual(analyzer.seed_glossary(store, result), 2)
            school = store.get_term("学校")
            self.assertIsNotNone(school)
            assert school is not None
            self.assertEqual(school.type, "term")
            store.close()

    def test_truncated_analysis_retries_with_a_larger_budget(self):
        analysis = {"style_guide": "保持克制。", "characters": [], "terms": []}
        calls = []

        def handler(messages, tier, json_mode):
            calls.append(1)
            if len(calls) == 1:
                raise TruncatedResponseError(
                    "OpenAI-compatible response was truncated at the token limit"
                )
            return json.dumps(analysis, ensure_ascii=False)

        client = FakeClient(handler=handler)
        result = Analyzer(client, _cfg()).analyze("……样章……")

        self.assertEqual(result["style_guide"], "保持克制。")
        self.assertEqual([call["max_tokens"] for call in client.calls], [8192, 12288])

    def test_persistent_truncation_fails_with_an_actionable_message(self):
        def handler(messages, tier, json_mode):
            raise TruncatedResponseError(
                "OpenAI-compatible response was truncated at the token limit"
            )

        client = FakeClient(handler=handler)
        with self.assertRaises(TruncatedResponseError) as raised:
            Analyzer(client, _cfg()).analyze("……样章……")

        self.assertEqual(len(client.calls), 3)
        self.assertEqual([call["max_tokens"] for call in client.calls], [8192, 12288, 16384])
        self.assertIn("max_output_tokens", str(raised.exception))

    def test_a_capped_profile_does_not_repeat_the_same_budget(self):
        """An explicit max_output_tokens wins over the hint, so every ladder step would send the
        same request; only the steps that actually raise the budget are worth retrying."""

        def handler(messages, tier, json_mode):
            raise TruncatedResponseError(
                "OpenAI-compatible response was truncated at the token limit"
            )

        cfg = _cfg()
        profile = cfg.llm.tiers["strong"]
        cfg.llm.models[profile] = cfg.llm.models[profile].model_copy(
            update={"max_output_tokens": 4096}
        )
        client = FakeClient(handler=handler)
        with self.assertRaises(TruncatedResponseError):
            Analyzer(client, cfg).analyze("……样章……")

        self.assertEqual([call["max_tokens"] for call in client.calls], [4096])

    def test_numeric_collections_are_normalized_to_empty_lists(self):
        analysis = {
            "genre": "novel",
            "characters": 3,
            "terms": 1.5,
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(analysis))

        with self.assertLogs("wenyi_core.agents.base", level="WARNING") as logs:
            result = Analyzer(client, _cfg()).analyze("sample")

        self.assertEqual(result["characters"], [])
        self.assertEqual(result["terms"], [])
        self.assertEqual(len(logs.records), 2)
        for record, field, actual_type in zip(
            logs.records, ("characters", "terms"), ("int", "float")
        ):
            self.assertIn("operation=analysis.style", record.getMessage())
            self.assertIn(f"field={field}", record.getMessage())
            self.assertIn(f"got {actual_type}", record.getMessage())


class TestExtractor(unittest.TestCase):
    def test_numeric_terms_collection_is_ignored(self):
        for value in (3, 1.5):
            with self.subTest(value=value):
                response = json.dumps({"terms": value})
                extractor = GlossaryExtractor(
                    FakeClient(handler=lambda m, t, j, response=response: response),
                    _cfg(),
                )

                with self.assertLogs("wenyi_core.agents.base", level="WARNING") as logs:
                    self.assertEqual(extractor.extract("source", "target", []), [])
                self.assertEqual(len(logs.records), 1)
                self.assertIn("operation=glossary.extract", logs.records[0].getMessage())
                self.assertIn("field=terms", logs.records[0].getMessage())

    def test_invalid_history_collection_warns_and_defers_unresolved_terms(self):
        extractor = GlossaryExtractor(
            FakeClient(handler=lambda m, t, j: json.dumps({"terms": 3})), _cfg()
        )
        term = GlossaryTerm(source="term", target="candidate")
        occurrences = {
            "term": TranslatedSegmentEvidence(
                chapter=0, segment=1, source="term", target="existing translation"
            )
        }
        with self.assertLogs("wenyi_core.agents.base", level="WARNING") as logs:
            result = extractor._align_with_first_occurrences([term], occurrences)

        self.assertEqual(result, ([], 0, 1))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("operation=glossary.align_history", logs.records[0].getMessage())
        self.assertIn("field=terms", logs.records[0].getMessage())

    def test_existing_context_only_includes_terms_repeated_in_source_corpus(self):
        prompts_seen: list[str] = []

        def handler(messages, tier, json_mode):
            prompts_seen.append(messages[-1]["content"])
            return json.dumps({"terms": []}, ensure_ascii=False)

        extractor = GlossaryExtractor(FakeClient(handler=handler), _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            store.upsert_term(GlossaryTerm(source="唯一术语", target="唯一译法"))
            store.upsert_term(GlossaryTerm(source="重复术语", target="重复译法"))

            extractor.extract_and_store(
                store,
                "本批原文。",
                "本批译文。",
                chapter=0,
                source_corpus="唯一术语只出现一次。重复术语出现，然后重复术语再次出现。",
            )

            self.assertEqual(len(store.all_terms()), 2)
            store.close()

        self.assertEqual(len(prompts_seen), 1)
        self.assertNotIn("唯一术语 → 唯一译法", prompts_seen[0])
        self.assertIn("重复术语 → 重复译法", prompts_seen[0])

    def test_extract_and_store(self):
        terms = {
            "terms": [
                {
                    "source": "堀北",
                    "target": "堀北",
                    "type": "person",
                    "gender": "female",
                    "aliases": ["堀北さん"],
                },
                {"source": "屋上", "target": "天台", "type": "place", "gender": "unknown"},
            ]
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(terms, ensure_ascii=False))
        ext = GlossaryExtractor(client, _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary = ext.extract_and_store(
                store, "堀北は屋上にいた。", "堀北在天台上。", chapter=1
            )
            self.assertEqual(summary["inserted"], 2)
            horikita = store.get_term("堀北")
            self.assertIsNotNone(horikita)
            assert horikita is not None
            self.assertEqual(horikita.gender, "female")
            self.assertEqual(horikita.aliases, ["堀北さん"])
            self.assertEqual(horikita.first_chapter, 1)
            # Normalize unknown gender to an empty value.
            rooftop = store.get_term("屋上")
            self.assertIsNotNone(rooftop)
            assert rooftop is not None
            self.assertEqual(rooftop.gender, "")
            store.close()

    def test_candidates_absent_from_the_batch_text_are_dropped(self):
        """A source the model paraphrased can never match text, so it is not stored.

        One real run stored 22 such rows: Chinese spellings of Japanese sources, mixed-script
        corruptions, misspellings and invented sentence fragments.
        """
        terms = {
            "terms": [
                {"source": "マセラティ", "target": "玛莎拉蒂", "type": "organization"},
                {"source": "マセラ蒂", "target": "玛莎拉蒂", "type": "organization"},
                {"source": "游泳池妄想", "target": "游泳池妄想", "type": "term"},
                {"source": "データが不足しているのだ。", "target": "数据不足。", "type": "term"},
            ]
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(terms, ensure_ascii=False))
        extractor = GlossaryExtractor(client, _cfg())

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary = extractor.extract_and_store(
                store,
                "呪われたマセラティに乗ってプール妄想を語った。",
                "坐着被诅咒的玛莎拉蒂，谈起了泳池妄想。",
                chapter=33,
            )
            self.assertEqual(summary["source_not_in_text"], 3)
            self.assertIsNotNone(store.get_term("マセラティ"))
            self.assertIsNone(store.get_term("マセラ蒂"))
            self.assertIsNone(store.get_term("游泳池妄想"))
            store.close()

    def test_malformed_optional_fields_fall_back_safely(self):
        terms = {
            "terms": [
                {
                    "source": "term",
                    "target": "术语",
                    "type": {"bad": 1},
                    "gender": ["bad"],
                    "aliases": 1,
                    "note": {"bad": 1},
                }
            ]
        }
        extractor = GlossaryExtractor(FakeClient(handler=lambda m, t, j: json.dumps(terms)), _cfg())

        result = extractor.extract("term", "术语", [])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "term")
        self.assertEqual(result[0].gender, "")
        self.assertEqual(result[0].aliases, [])
        self.assertEqual(result[0].note, "")

    def test_multiple_new_terms_use_targets_from_first_translated_occurrences(self):
        calls: list[str] = []

        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            calls.append(system)
            if "terminology consistency aligner" in system:
                user = messages[-1]["content"]
                self.assertIn("綾小路第一次走进教室。", user)
                self.assertIn("绫小路第一次走进教室。", user)
                self.assertIn("堀北站在窗边。", user)
                self.assertIn('"proposed_target": "掘北"', user)
                return json.dumps(
                    {
                        "terms": [
                            {"source": "綾小路", "target": "绫小路"},
                            {"source": "堀北", "target": "堀北"},
                        ]
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "terms": [
                        {"source": "綾小路", "target": "凌小路", "type": "person"},
                        {"source": "堀北", "target": "掘北", "type": "person"},
                    ]
                },
                ensure_ascii=False,
            )

        extractor = GlossaryExtractor(FakeClient(handler=handler), _cfg())
        history = [
            TranslatedSegmentEvidence(
                chapter=0,
                segment=3,
                source="綾小路第一次走进教室。",
                target="绫小路第一次走进教室。",
            ),
            TranslatedSegmentEvidence(
                chapter=0,
                segment=4,
                source="堀北站在窗边。",
                target="堀北站在窗边。",
            ),
        ]
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary = extractor.extract_and_store(
                store,
                "后来綾小路和堀北再次出现。",
                "后来凌小路和掘北再次出现。",
                chapter=2,
                history=history,
                before=(2, 0),
            )
            ayanokoji = store.get_term("綾小路")
            horikita = store.get_term("堀北")
            self.assertIsNotNone(ayanokoji)
            self.assertIsNotNone(horikita)
            assert ayanokoji is not None
            assert horikita is not None
            self.assertEqual(ayanokoji.target, "绫小路")
            self.assertEqual(horikita.target, "堀北")
            self.assertEqual(ayanokoji.first_chapter, 0)
            self.assertEqual(horikita.first_chapter, 0)
            self.assertEqual(summary["history_matched"], 2)
            self.assertEqual(summary["history_aligned"], 2)
            self.assertEqual(summary["history_unresolved"], 0)
            store.close()
        self.assertEqual(len(calls), 2)

    def test_stored_terms_keep_conflicts_for_resolution(self):
        """A proposal against a stored mapping is recorded, never settled by the extractor.

        The restricted resolve only covers a mapping the same batch inserted: alignment history
        is computed before the write and holds only terms that were not stored yet, so a stored
        mapping always waits for a human or the terminology arbiter.
        """
        terms = {"terms": [{"source": "いるかホテル", "target": "海豚酒店", "type": "place"}]}
        client = FakeClient(handler=lambda m, t, j: json.dumps(terms, ensure_ascii=False))
        extractor = GlossaryExtractor(client, _cfg())

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            store.upsert_term(GlossaryTerm(source="いるかホテル", target="海豚旅店"), chapter=0)
            summary = extractor.extract_and_store(
                store,
                "いるかホテルへ行く。",
                "去海豚酒店。",
                chapter=1,
                history=[
                    TranslatedSegmentEvidence(
                        chapter=0,
                        segment=0,
                        source="いるかホテルへ行く。",
                        target="去海豚旅店。",
                    )
                ],
                before=(1, 0),
                source_corpus="いるかホテルへ行く。いるかホテルは古い。",
            )
            term = store.get_term("いるかホテル")
            assert term is not None
            self.assertEqual(summary["conflict"], 1)
            self.assertEqual(summary["auto_locked"], 0)
            # The established mapping stands and the disagreement stays for a decision.
            self.assertEqual(term.target, "海豚旅店")
            self.assertEqual(term.status, "conflict")
            self.assertEqual(len(store.open_conflicts()), 1)
            store.close()

    def test_new_term_without_prior_occurrence_is_inserted_directly(self):
        terms = {"terms": [{"source": "綾小路", "target": "绫小路", "type": "person"}]}
        client = FakeClient(handler=lambda m, t, j: json.dumps(terms, ensure_ascii=False))
        extractor = GlossaryExtractor(client, _cfg())

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary = extractor.extract_and_store(
                store,
                "綾小路第一次出现。",
                "绫小路第一次出现。",
                chapter=0,
                history=[],
                before=(0, 0),
            )
            self.assertIsNotNone(store.get_term("綾小路"))
            self.assertEqual(summary["history_matched"], 0)
            store.close()

        self.assertEqual(len(client.calls), 1)

    def test_unresolved_historical_term_is_not_locked_to_later_translation(self):
        responses = iter(
            [
                json.dumps(
                    {"terms": [{"source": "綾小路", "target": "凌小路"}]},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"terms": [{"source": "綾小路", "target": ""}]},
                    ensure_ascii=False,
                ),
            ]
        )
        extractor = GlossaryExtractor(FakeClient(handler=lambda m, t, j: next(responses)), _cfg())
        history = [
            TranslatedSegmentEvidence(
                chapter=0,
                segment=0,
                source="綾小路がいた。",
                target="他在那里。",
            )
        ]

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary = extractor.extract_and_store(
                store,
                "綾小路が戻った。",
                "凌小路回来了。",
                chapter=1,
                history=history,
                before=(1, 0),
            )
            self.assertIsNone(store.get_term("綾小路"))
            self.assertEqual(summary["history_unresolved"], 1)
            store.close()


class TestRollingContext(unittest.TestCase):
    def test_render_and_bound(self):
        ctx = RollingContext(max_recent_keep=3)
        ctx.add_targets(["a", "b", "c", "d", "e"])
        self.assertEqual(ctx.recent_targets, ["c", "d", "e"])  # Bound retained context length.
        rendered = ctx.render(n_recent=2)  # Use only the two most recent paragraphs.
        self.assertIn("d", rendered)
        self.assertIn("e", rendered)
        self.assertNotIn("c", rendered)

    def test_add_pairs_skips_empty_targets_and_renders_source_translation(self):
        ctx = RollingContext(max_recent_keep=3)
        ctx.add_pairs(
            ["s1", "s2", "s3", "s4"],
            ["t1", "", "  ", "t4"],
        )
        self.assertEqual(
            ctx.recent_pairs, [{"source": "s1", "target": "t1"}, {"source": "s4", "target": "t4"}]
        )
        self.assertEqual(ctx.recent_targets, ["t1", "t4"])
        rendered = ctx.render(2, with_source=True)
        self.assertIn("Source: s1", rendered)
        self.assertIn("Translation: t1", rendered)
        self.assertIn("Source: s4", rendered)
        self.assertIn("Translation: t4", rendered)
        targets_only = ctx.render(2, with_source=False)
        self.assertEqual(targets_only, "t1\nt4")

    def test_roundtrip(self):
        ctx = RollingContext(recent_targets=["x", "y"], max_recent_keep=75)
        ctx2 = RollingContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx2.recent_targets, ["x", "y"])
        self.assertEqual(ctx2.max_recent_keep, 75)

    def test_pairs_roundtrip_writes_recent_targets_for_compat(self):
        ctx = RollingContext(max_recent_keep=75)
        ctx.add_pairs(["s1", "s2"], ["t1", "t2"])
        payload = ctx.to_dict()
        self.assertEqual(payload["recent_targets"], ["t1", "t2"])
        self.assertEqual(
            payload["recent_pairs"],
            [{"source": "s1", "target": "t1"}, {"source": "s2", "target": "t2"}],
        )
        restored = RollingContext.from_dict(payload)
        self.assertEqual(restored.recent_pairs, ctx.recent_pairs)
        self.assertEqual(restored.recent_targets, ["t1", "t2"])

    def test_legacy_recent_targets_json_loads_as_target_only(self):
        restored = RollingContext.from_dict(
            {"recent_targets": ["旧译A", "旧译B"], "max_recent_keep": 40}
        )
        self.assertEqual(restored.recent_pairs, [])
        self.assertEqual(restored.recent_targets, ["旧译A", "旧译B"])
        rendered = restored.render(2, with_source=True)
        self.assertEqual(rendered, "旧译A\n旧译B")
        self.assertNotIn("Source:", rendered)

    def test_configured_minimum_expands_saved_context_limit(self):
        ctx = RollingContext.from_dict(
            {"recent_targets": [str(i) for i in range(40)]},
            min_recent_keep=100,
        )
        self.assertEqual(ctx.max_recent_keep, 100)

    def test_sync_context_chapter_prefix_overrides_trailing_pairs(self):
        from wenyi_core.ingest.models import Segment
        from wenyi_core.pipeline.translation import TranslationService

        ctx = RollingContext(max_recent_keep=10)
        ctx.add_pairs(["old1", "old2", "old3"], ["译1", "译2", "译3"])
        segments = [
            Segment(index=0, source="新源0", target="新译0"),
            Segment(index=1, source="新源1", target="新译1"),
            Segment(index=2, source="新源2", target="新译2"),
        ]
        TranslationService.sync_context_chapter_prefix(ctx, segments, 3)
        self.assertEqual(
            ctx.recent_pairs,
            [
                {"source": "新源0", "target": "新译0"},
                {"source": "新源1", "target": "新译1"},
                {"source": "新源2", "target": "新译2"},
            ],
        )
        self.assertEqual(ctx.recent_targets, ["新译0", "新译1", "新译2"])


if __name__ == "__main__":
    unittest.main()
