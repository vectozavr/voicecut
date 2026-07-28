from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock


from voicecut.common import (
    Candidate,
    ObservedWord,
    ScriptPhrase,
    parse_script,
    repeated_surplus,
    split_long_clause,
    tokenize,
)
from voicecut.plan import (
    candidate_for_span,
    has_text_equivalent_alternative,
    nontrivial_alternative_margin,
    path_is_globally_monotonic,
    refine_retry_path,
    retry_repair_review_reasons,
    selection_quality_review_reasons,
    select_global_path,
    top_candidates,
)
from voicecut.qa import (
    exact_surplus_repeats,
    restart_insertions,
    single_token_disfluencies,
)
from voicecut.transcribe_faster import segment_rejection_reasons


def phrase(index: int, text: str) -> ScriptPhrase:
    return ScriptPhrase(
        phrase_index=index,
        unit_index=index,
        phrase_in_unit=0,
        line_number=index + 1,
        paragraph_index=index + 1,
        cue_before=index == 0,
        text=text,
        tokens=tokenize(text),
        pause_after="sentence",
    )


def observed_words(text: str, step: float = 0.2) -> list[ObservedWord]:
    words: list[ObservedWord] = []
    for index, raw_word in enumerate(text.split()):
        words.append(
            ObservedWord(
                word_index=index,
                atom_index=0,
                text=raw_word,
                tokens=tokenize(raw_word),
                start=index * step,
                end=index * step + step * 0.75,
                probability=0.95,
            )
        )
    return words


def candidate(
    phrase_index: int,
    word_start: int,
    word_end: int,
    score: float,
) -> Candidate:
    return Candidate(
        phrase_index=phrase_index,
        word_start=word_start,
        word_end=word_end,
        score=score,
        f1=score,
        recall=score,
        precision=score,
        edit_similarity=score,
        prefix_recall=score,
        suffix_recall=score,
        mean_probability=0.95,
        repeated_surplus=0.0,
        max_gap=0.05,
        transcript=f"candidate-{phrase_index}-{word_start}",
        source_start=word_start * 0.1,
        source_end=word_end * 0.1,
    )


class TokenizationAndRepeatTests(unittest.TestCase):
    def test_tokenization_normalizes_contractions_numbers_and_inflections(self) -> None:
        self.assertEqual(
            tokenize("Weights aren't removed—50% sparsity."),
            ["weight", "are", "not", "remov", "fifty", "percent", "sparsity"],
        )

    def test_tokenization_preserves_timed_percent_and_once_homophone(self) -> None:
        self.assertEqual(tokenize("%"), ["percent"])
        self.assertEqual(tokenize("Once"), tokenize("1"))

    def test_retry_split_prefers_a_real_clause_boundary(self) -> None:
        self.assertEqual(
            split_long_clause(
                "In this example, “Once” has the highest probability,",
                {},
                5,
            ),
            [
                "In this example,",
                "“Once” has the highest probability,",
            ],
        )

    def test_intentional_script_repeat_is_not_surplus(self) -> None:
        reference = tokenize("This is very very important.")
        self.assertEqual(repeated_surplus(reference, reference), 0.0)
        self.assertGreater(
            repeated_surplus(reference, tokenize("This is very very very important.")),
            0.0,
        )

    def test_exact_repeat_detector_respects_script_repeat_count(self) -> None:
        reference = tokenize("Try again, try again, and continue.")
        times = [(index * 0.1, index * 0.1 + 0.08) for index in range(len(reference))]
        self.assertEqual(exact_surplus_repeats(reference, reference, times), [])

        observed = tokenize("Try again, try again, try again, and continue.")
        observed_times = [
            (index * 0.1, index * 0.1 + 0.08) for index in range(len(observed))
        ]
        findings = exact_surplus_repeats(reference, observed, observed_times)
        self.assertTrue(
            any(
                "try again" in finding["phrase"]
                and finding["observed_count"] > finding["script_count"]
                for finding in findings
            )
        )

    def test_known_opening_restart_is_a_hard_failure(self) -> None:
        reference = tokenize(
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with the familiar words."
        )
        observed = tokenize(
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with familiar with the familiar words."
        )
        times = [(index * 0.2, index * 0.2 + 0.15) for index in range(len(observed))]
        findings = restart_insertions(reference, observed, times)
        self.assertTrue(findings)
        self.assertTrue(
            any(
                finding["severity"] == "fail"
                and finding["observed_fragment"] == "familiar with"
                and finding["two_sided_restart"]
                for finding in findings
            )
        )

    def test_restart_detector_never_compares_an_insertion_with_itself(self) -> None:
        reference = tokenize("We introduce the method we developed.")
        observed = tokenize(
            "We introduce the method we developed do not forget to subscribe."
        )
        times = [(index * 0.1, index * 0.1 + 0.08) for index in range(len(observed))]
        self.assertEqual(restart_insertions(reference, observed, times), [])

    def test_independent_asr_rejects_implausible_outro_hallucination(self) -> None:
        segment = {
            "start": 28.14,
            "end": 28.50,
            "avg_logprob": -0.85,
            "no_speech_prob": 0.30,
            "words": [
                {
                    "start": 28.38,
                    "end": 28.38,
                    "word": word,
                    "probability": probability,
                }
                for word, probability in [
                    ("forget", 0.12),
                    ("to", 0.84),
                    ("subscribe", 0.61),
                    ("to", 0.46),
                    ("our", 0.41),
                    ("channel", 0.79),
                ]
            ],
        }
        reasons = segment_rejection_reasons(segment)
        self.assertIn("implausible_word_timing", reasons)
        self.assertIn("low_confidence_non_speech", reasons)

    def test_single_word_stutter_and_filler_cannot_hide_in_long_audio(self) -> None:
        reference = tokenize("We remove the selected weight and continue.")
        stuttered = tokenize("We remove the the selected weight and continue.")
        stutter_times = [
            (index * 0.1, index * 0.1 + 0.08)
            for index in range(len(stuttered))
        ]
        stutter_findings = single_token_disfluencies(
            reference, stuttered, stutter_times
        )
        self.assertTrue(
            any(
                finding["kind"] == "single_token_stutter"
                for finding in stutter_findings
            )
        )

        with_filler = tokenize("We remove uh the selected weight and continue.")
        filler_times = [
            (index * 0.1, index * 0.1 + 0.08)
            for index in range(len(with_filler))
        ]
        filler_findings = single_token_disfluencies(
            reference, with_filler, filler_times
        )
        self.assertTrue(
            any(
                finding["kind"] == "surplus_spoken_filler"
                and finding["severity"] == "fail"
                for finding in filler_findings
            )
        )

        intentional = tokenize("This is very very important.")
        intentional_times = [
            (index * 0.1, index * 0.1 + 0.08)
            for index in range(len(intentional))
        ]
        self.assertEqual(
            single_token_disfluencies(
                intentional, intentional, intentional_times
            ),
            [],
        )

    def test_hard_wrapped_lines_remain_one_paragraph(self) -> None:
        markdown = (
            "This sentence is hard wrapped\n"
            "across two physical lines and ends here.\n"
            "A second sentence shares the paragraph.\n"
            "\n"
            "This starts a new paragraph.\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "script.md"
            script_path.write_text(markdown, encoding="utf-8")
            units = parse_script(script_path)

        self.assertEqual(len(units), 3)
        self.assertEqual(
            units[0].text,
            "This sentence is hard wrapped across two physical lines and ends here.",
        )
        self.assertEqual(
            [unit.paragraph_index for unit in units],
            [1, 1, 2],
        )
        self.assertEqual([unit.cue_before for unit in units], [True, False, True])

    def test_resource_links_and_fenced_blocks_are_not_narration(self) -> None:
        markdown = (
            "# Closing\n\n"
            "Thank you for listening.\n\n"
            "Code: [example/tool](https://github.com/example/tool)\n\n"
            "```\n"
            "voicecut --audio raw.wav --script script.md\n"
            "```\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "script.md"
            script_path.write_text(markdown, encoding="utf-8")
            units = parse_script(script_path)

        self.assertEqual([unit.text for unit in units], ["Thank you for listening."])


class TakeSelectionTests(unittest.TestCase):
    def test_global_path_is_monotonic_and_can_reject_greedy_choice(self) -> None:
        phrases = [phrase(0, "First phrase."), phrase(1, "Second phrase.")]
        early_first = candidate(0, 0, 3, 0.89)
        locally_best_but_too_late = candidate(0, 10, 13, 1.0)
        only_second = candidate(1, 4, 9, 0.99)

        selected = select_global_path(
            phrases,
            [[early_first, locally_best_but_too_late], [only_second]],
            allowed_missing_units=set(),
        )

        self.assertIs(selected[0], early_first)
        self.assertIs(selected[1], only_second)
        nonmissing = [item for item in selected if item is not None]
        self.assertTrue(
            all(
                left.word_end <= right.word_start
                for left, right in zip(nonmissing, nonmissing[1:])
            )
        )

    def test_synthetic_retry_prefers_clean_later_take(self) -> None:
        script = (
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with the familiar words"
        )
        broken = (
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with familiar with the familiar words"
        )
        bridge = "well let me start again"
        raw = f"{broken} {bridge} {script}"
        words = observed_words(raw)
        target = phrase(0, script)

        broken_end = len(broken.split())
        clean_start = broken_end + len(bridge.split())
        clean_end = clean_start + len(script.split())
        broken_candidate = candidate_for_span(
            target, words, 0, broken_end, 0, len(words)
        )
        clean_candidate = candidate_for_span(
            target, words, clean_start, clean_end, 0, len(words)
        )
        ranked = top_candidates(target, words, 0, len(words), limit=10)

        self.assertGreater(broken_candidate.repeated_surplus, 0.0)
        self.assertEqual(clean_candidate.repeated_surplus, 0.0)
        self.assertGreater(clean_candidate.score, broken_candidate.score)
        self.assertEqual(ranked[0].word_start, clean_start)
        self.assertEqual(ranked[0].word_end, clean_end)
        self.assertNotIn("familiar with the familiar", ranked[0].transcript.lower())

    def test_retry_refinement_replaces_bad_opening_with_two_clean_subspans(
        self,
    ) -> None:
        script = (
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with the familiar words."
        )
        broken = (
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with familiar with the familiar words."
        )
        words = observed_words(broken)
        target = phrase(0, script)
        coarse = candidate_for_span(target, words, 0, len(words), 0, len(words))

        refined_phrases, _, refined_path, repairs = refine_retry_path(
            [target],
            [[coarse]],
            [coarse],
            words,
            {},
            set(),
            candidate_limit=14,
        )

        self.assertEqual(len(refined_phrases), 2)
        self.assertEqual(len(repairs), 1)
        self.assertTrue(all(selected is not None for selected in refined_path))
        selected = [item for item in refined_path if item is not None]
        self.assertEqual(
            [(item.word_start, item.word_end) for item in selected],
            [(0, 18), (20, 23)],
        )
        self.assertLessEqual(selected[0].word_end, selected[1].word_start)
        repaired_text = " ".join(item.transcript for item in selected).lower()
        self.assertNotIn("familiar with the familiar", repaired_text)
        self.assertEqual(repairs[0]["repaired_repeated_surplus"], 0.0)
        self.assertEqual(repairs[0]["status"], "accepted")
        self.assertTrue(path_is_globally_monotonic(refined_path))
        self.assertTrue(
            all(
                selection_quality_review_reasons(subphrase, selected_candidate) == []
                for subphrase, selected_candidate in zip(
                    refined_phrases,
                    selected,
                )
            )
        )

    def test_retry_refinement_leaves_clean_coarse_candidate_unchanged(self) -> None:
        script = (
            "Give a large language model a simple instruction such as write a "
            "story and it may begin with the familiar words."
        )
        words = observed_words(script)
        target = phrase(0, script)
        coarse = candidate_for_span(target, words, 0, len(words), 0, len(words))

        refined_phrases, refined_sets, refined_path, repairs = refine_retry_path(
            [target],
            [[coarse]],
            [coarse],
            words,
            {},
            set(),
            candidate_limit=14,
        )

        self.assertEqual(refined_phrases, [target])
        self.assertEqual(refined_sets, [[coarse]])
        self.assertEqual(refined_path, [coarse])
        self.assertEqual(repairs, [])

    def test_adjacent_repairs_cannot_expand_into_the_same_gap(self) -> None:
        script = "word word word word word word word word"
        phrases = [phrase(0, script), phrase(1, script)]
        words = observed_words(" ".join(["word"] * 60))
        first_coarse = candidate(0, 0, 10, 0.95)
        second_coarse = candidate(1, 30, 40, 0.95)
        for coarse in (first_coarse, second_coarse):
            coarse.repeated_surplus = 2.0
            coarse.transcript = " ".join(["word"] * 10)

        observed_search_starts: dict[int, set[int]] = {0: set(), 1: set()}

        def fake_top_candidates(
            subphrase: ScriptPhrase,
            _: list[ObservedWord],
            search_start: int,
            search_end: int,
            *,
            limit: int,
        ) -> list[Candidate]:
            del search_end, limit
            observed_search_starts[subphrase.unit_index].add(search_start)
            if subphrase.unit_index == 0:
                start = 20 + 4 * subphrase.phrase_in_unit
            else:
                # With the old coarse-path fence this starts at 22 and
                # overlaps the first repair.  The refined-path fence is 28.
                base = 28 if search_start >= 28 else 22
                start = base + 4 * subphrase.phrase_in_unit
            result = candidate(
                subphrase.phrase_index,
                start,
                start + 4,
                0.99,
            )
            result.transcript = "word word word word"
            return [result]

        with mock.patch(
            "voicecut.plan.top_candidates",
            side_effect=fake_top_candidates,
        ):
            refined_phrases, _, refined_path, repairs = refine_retry_path(
                phrases,
                [[first_coarse], [second_coarse]],
                [first_coarse, second_coarse],
                words,
                {},
                set(),
                candidate_limit=8,
            )

        selected = [item for item in refined_path if item is not None]
        self.assertEqual(len(refined_phrases), 4)
        self.assertEqual(
            [(item.word_start, item.word_end) for item in selected],
            [(20, 24), (24, 28), (28, 32), (32, 36)],
        )
        self.assertIn(28, observed_search_starts[1])
        self.assertTrue(path_is_globally_monotonic(refined_path))
        self.assertEqual(
            [record["status"] for record in repairs],
            ["accepted", "accepted"],
        )

    def test_weak_subphrase_repair_stays_coarse_and_requires_review(self) -> None:
        script = "word word word word word word word word"
        target = phrase(0, script)
        words = observed_words(" ".join(["word"] * 30))
        coarse = candidate(0, 0, 10, 0.95)
        coarse.repeated_surplus = 2.0
        coarse.transcript = " ".join(["word"] * 10)

        def weak_candidates(
            subphrase: ScriptPhrase,
            _: list[ObservedWord],
            search_start: int,
            search_end: int,
            *,
            limit: int,
        ) -> list[Candidate]:
            del search_start, search_end, limit
            start = 10 + 4 * subphrase.phrase_in_unit
            result = candidate(
                subphrase.phrase_index,
                start,
                start + 4,
                0.95,
            )
            result.transcript = "word word word word"
            if subphrase.phrase_in_unit == 1:
                result.recall = 0.82
            return [result]

        with mock.patch(
            "voicecut.plan.top_candidates",
            side_effect=weak_candidates,
        ):
            refined_phrases, refined_sets, refined_path, repairs = refine_retry_path(
                [target],
                [[coarse]],
                [coarse],
                words,
                {},
                set(),
                candidate_limit=8,
            )

        self.assertEqual(refined_phrases, [target])
        self.assertEqual(refined_sets, [[coarse]])
        self.assertEqual(refined_path, [coarse])
        self.assertEqual(repairs[0]["status"], "rejected")
        self.assertEqual(repairs[0]["reason"], "insufficient_repair_quality")
        self.assertEqual(
            retry_repair_review_reasons(target, coarse, repairs),
            ["unsafe_retry_repair_rejected"],
        )

    def test_repair_cannot_degrade_combined_coarse_recall(self) -> None:
        script = "word word word word word word word word"
        target = phrase(0, script)
        words = observed_words(" ".join(["word"] * 30))
        coarse = candidate(0, 0, 10, 0.95)
        coarse.repeated_surplus = 2.0
        coarse.transcript = " ".join(["word"] * 10)

        def degraded_candidates(
            subphrase: ScriptPhrase,
            _: list[ObservedWord],
            search_start: int,
            search_end: int,
            *,
            limit: int,
        ) -> list[Candidate]:
            del search_start, search_end, limit
            start = 10 + 4 * subphrase.phrase_in_unit
            end = start + (4 if subphrase.phrase_in_unit == 0 else 3)
            result = candidate(
                subphrase.phrase_index,
                start,
                end,
                0.95,
            )
            result.transcript = " ".join(["word"] * (end - start))
            return [result]

        with mock.patch(
            "voicecut.plan.top_candidates",
            side_effect=degraded_candidates,
        ):
            _, _, refined_path, repairs = refine_retry_path(
                [target],
                [[coarse]],
                [coarse],
                words,
                {},
                set(),
                candidate_limit=8,
            )

        self.assertEqual(refined_path, [coarse])
        self.assertEqual(repairs[0]["status"], "rejected")
        self.assertLess(repairs[0]["repaired_recall"], coarse.recall)

    def test_final_selection_quality_floors_are_fail_closed(self) -> None:
        short = phrase(0, "one two three four five six seven eight")
        weak = candidate(0, 0, 8, 0.84)
        weak.recall = 0.89
        weak.precision = 0.79
        weak.f1 = 0.84
        self.assertEqual(
            selection_quality_review_reasons(short, weak),
            [
                "low_script_recall",
                "low_take_precision",
                "low_alignment_f1",
            ],
        )
        clean = candidate(0, 0, 8, 0.95)
        self.assertEqual(selection_quality_review_reasons(short, clean), [])

    def test_equivalent_high_quality_take_is_not_semantically_ambiguous(self) -> None:
        selected = candidate(0, 10, 20, 0.950)
        selected.transcript = "The remaining parameters compensate for the removal."
        alternate = candidate(0, 40, 50, 0.944)
        alternate.transcript = "The remaining parameters compensate for the removal"
        candidates = [selected, alternate]

        margin = nontrivial_alternative_margin(selected, candidates)
        equivalent = has_text_equivalent_alternative(selected, candidates, {})

        self.assertLess(margin, 0.018)
        self.assertTrue(equivalent)
        self.assertFalse(margin < 0.018 and not equivalent)

    def test_different_competitive_take_remains_semantically_ambiguous(self) -> None:
        selected = candidate(0, 10, 20, 0.950)
        selected.transcript = "The remaining parameters compensate for the removal."
        alternate = candidate(0, 40, 50, 0.944)
        alternate.transcript = "The removed parameters cannot compensate for the update."
        candidates = [selected, alternate]

        margin = nontrivial_alternative_margin(selected, candidates)
        equivalent = has_text_equivalent_alternative(selected, candidates, {})

        self.assertLess(margin, 0.018)
        self.assertFalse(equivalent)
        self.assertTrue(margin < 0.018 and not equivalent)


if __name__ == "__main__":
    unittest.main()
