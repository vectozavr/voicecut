from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from voicecut.common import read_json, write_json
from voicecut.streaming_narration import (
    DecisionValidationError,
    SourceGroundingValidationError,
    TranscriptWord,
    build_conservative_delivery_plan,
    load_transcript_words,
    repair_plan_for_acoustic_safety,
    run_streaming_planner,
    streaming_response_schema,
    validate_decision,
)


class FakeStreamingBackend:
    backend_name = "fake"
    model = "deterministic-test-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.request_ids: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        self.prompts.append(prompt)
        self.request_ids.append(request_id)
        if not response_schema:
            raise AssertionError("the backend did not receive a JSON schema")
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def response(
    *,
    finalized: list[dict[str, Any]],
    pending_start_word_id: int | None,
    pending_reason: str = "test decision",
) -> str:
    return json.dumps(
        {
            "finalized": finalized,
            "pending_start_word_id": pending_start_word_id,
            "pending_reason": pending_reason,
        }
    )


def thought(
    text: str,
    *ranges: tuple[int, int, str, str, str],
) -> dict[str, Any]:
    return {
        "canonical_text": text,
        "source_ranges": [
            {
                "first_word_id": first_word_id,
                "last_word_id": last_word_id,
                "first_word": first_word,
                "last_word": last_word,
                "canonical_text": range_text,
            }
            for (
                first_word_id,
                last_word_id,
                first_word,
                last_word,
                range_text,
            ) in ranges
        ],
    }


def transcript_with_word_groups(
    groups: list[tuple[float, list[str]]],
) -> dict[str, Any]:
    atoms = []
    for atom_index, (start, texts) in enumerate(groups):
        atoms.append(
            {
                "atom_index": atom_index,
                "words": [
                    {
                        "word": text,
                        "start": start + word_index,
                        "end": start + word_index + 0.4,
                    }
                    for word_index, text in enumerate(texts)
                ],
            }
        )
    return {"atoms": atoms}


class StreamingNarrationTests(unittest.TestCase):
    def test_conservative_delivery_preserves_audio_across_unsafe_cut(self) -> None:
        words = [
            {"id": 0, "text": "Alpha", "start": 0.0, "end": 0.2},
            {"id": 1, "text": "wrong", "start": 0.2, "end": 0.4},
            {"id": 2, "text": "Beta", "start": 0.4, "end": 0.6},
        ]
        plan = {
            "status": "complete",
            "words": words,
            "committed": [
                {
                    "canonical_text": "Alpha",
                    "source_ranges": [
                        {
                            "start_word_id": 0,
                            "end_word_id": 1,
                            "first_word_id": 0,
                            "last_word_id": 0,
                            "first_word": "Alpha",
                            "last_word": "Alpha",
                            "canonical_text": "Alpha",
                        }
                    ],
                },
                {
                    "canonical_text": "Beta",
                    "source_ranges": [
                        {
                            "start_word_id": 2,
                            "end_word_id": 3,
                            "first_word_id": 2,
                            "last_word_id": 2,
                            "first_word": "Beta",
                            "last_word": "Beta",
                            "canonical_text": "Beta",
                        }
                    ],
                },
            ],
        }
        unsafe = {
            "status": "unsafe",
            "source_intervals": [
                {
                    "start_word_id": 0,
                    "end_word_id": 1,
                    "merged_original_ranges": [{"thought_index": 0}],
                },
                {
                    "start_word_id": 2,
                    "end_word_id": 3,
                    "merged_original_ranges": [{"thought_index": 1}],
                },
            ],
            "boundaries": [
                {
                    "boundary_id": "range_0000_end",
                    "safety_status": "mfa_word_mapping_failed",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "streaming_plan.json"
            unsafe_path = root / "unsafe_boundary_plan.json"
            write_json(plan_path, plan)
            write_json(unsafe_path, unsafe)

            fallback = build_conservative_delivery_plan(
                plan_path=plan_path,
                boundary_plan_path=unsafe_path,
                output_dir=root / "fallback",
            )

            first = fallback["committed"][0]
            self.assertEqual(first["canonical_text"], "Alpha wrong")
            self.assertEqual(
                first["source_ranges"][0]["end_word_id"],
                2,
            )
            self.assertEqual(
                fallback["delivery_fallback"]["status"],
                "complete_with_preserved_source_context",
            )
            grounding = read_json(root / "fallback" / "grounding_validation.json")
            self.assertEqual(grounding["status"], "valid")
            self.assertEqual(grounding["unsupported_tokens"], [])
            self.assertEqual(grounding["unrepresented_source_tokens"], [])

    def test_conservative_delivery_accepts_exact_punctuation_only_anchor(
        self,
    ) -> None:
        words = [
            {"id": 0, "text": "Alpha", "start": 0.0, "end": 0.2},
            {"id": 1, "text": "-", "start": 0.2, "end": 0.3},
            {"id": 2, "text": "Beta", "start": 0.4, "end": 0.6},
        ]
        plan = {
            "status": "complete",
            "words": words,
            "committed": [
                {
                    "canonical_text": "Alpha",
                    "source_ranges": [
                        {
                            "start_word_id": 0,
                            "end_word_id": 1,
                            "first_word_id": 0,
                            "last_word_id": 0,
                            "first_word": "Alpha",
                            "last_word": "Alpha",
                            "canonical_text": "Alpha",
                        }
                    ],
                },
                {
                    "canonical_text": "Beta",
                    "source_ranges": [
                        {
                            "start_word_id": 2,
                            "end_word_id": 3,
                            "first_word_id": 2,
                            "last_word_id": 2,
                            "first_word": "Beta",
                            "last_word": "Beta",
                            "canonical_text": "Beta",
                        }
                    ],
                },
            ],
        }
        unsafe = {
            "status": "unsafe",
            "source_intervals": [
                {
                    "start_word_id": 0,
                    "end_word_id": 1,
                    "merged_original_ranges": [{"thought_index": 0}],
                },
                {
                    "start_word_id": 2,
                    "end_word_id": 3,
                    "merged_original_ranges": [{"thought_index": 1}],
                },
            ],
            "boundaries": [
                {
                    "boundary_id": "range_0000_end",
                    "safety_status": "mfa_word_mapping_failed",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "streaming_plan.json"
            unsafe_path = root / "unsafe_boundary_plan.json"
            write_json(plan_path, plan)
            write_json(unsafe_path, unsafe)

            fallback = build_conservative_delivery_plan(
                plan_path=plan_path,
                boundary_plan_path=unsafe_path,
                output_dir=root / "fallback",
            )

            first = fallback["committed"][0]
            self.assertEqual(first["canonical_text"], "Alpha -")
            self.assertEqual(first["source_ranges"][0]["last_word"], "-")
            self.assertEqual(first["source_ranges"][0]["end_word_id"], 2)
            grounding = read_json(root / "fallback" / "grounding_validation.json")
            self.assertEqual(grounding["status"], "valid")
            self.assertEqual(grounding["unsupported_tokens"], [])
            self.assertEqual(grounding["unrepresented_source_tokens"], [])

    def test_acoustic_repair_reselects_thought_without_reusing_unsafe_cut(
        self,
    ) -> None:
        texts = [
            "And",
            "it",
            "might",
            "begin",
            "with",
            "familiar",
            "with",
            "the",
            "familiar",
            "words",
            "once",
            "upon",
            "a",
            "time.",
        ]
        transcript = transcript_with_word_groups([(0.0, texts)])
        initial_backend = FakeStreamingBackend(
            [
                response(finalized=[], pending_start_word_id=0),
                response(
                    finalized=[
                        thought(
                            "And it might begin with familiar words once upon a time.",
                            (
                                0,
                                5,
                                "And",
                                "familiar",
                                "And it might begin with familiar",
                            ),
                            (
                                9,
                                13,
                                "words",
                                "time.",
                                "words once upon a time.",
                            ),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )
        repair_backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "And it might begin with familiar words once upon a time.",
                            (
                                0,
                                5,
                                "And",
                                "familiar",
                                "And it might begin with familiar",
                            ),
                            (
                                9,
                                13,
                                "words",
                                "time.",
                                "words once upon a time.",
                            ),
                        )
                    ],
                    pending_start_word_id=None,
                ),
                response(
                    finalized=[
                        thought(
                            "And it might begin with the familiar words once upon "
                            "a time.",
                            (
                                0,
                                3,
                                "And",
                                "begin",
                                "And it might begin",
                            ),
                            (
                                6,
                                13,
                                "with",
                                "time.",
                                "with the familiar words once upon a time.",
                            ),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            initial_dir = root / "initial"
            run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=initial_dir,
                backend=initial_backend,
            )
            boundary_path = root / "rejected_boundary_plan.json"
            write_json(
                boundary_path,
                {
                    "status": "unsafe",
                    "boundaries": [
                        {
                            "boundary_id": "range_0000_end",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "source_word_ids": {
                                "last_retained_left": 5,
                                "first_omitted": 6,
                                "last_omitted": 8,
                                "first_retained_right": 9,
                            },
                            "aligned_timestamps": {},
                            "forbidden_word_ids": [5],
                            "forbidden_source_edges": [
                                {
                                    "boundary_kind": "selected_to_omitted",
                                    "retained_word_id": 5,
                                    "omitted_word_id": 6,
                                }
                            ],
                            "failure_reason": "weak_terminal_word_support",
                            "retained_word_support": {
                                "word_id": 5,
                                "source_text": "familiar",
                                "status": "weak_terminal_word_support",
                                "edge_character_scores": [0.85, 0.41, 0.49],
                                "minimum_edge_score": 0.41,
                                "local_context_median_score": 0.91,
                                "edge_to_context_score_ratio": 0.45,
                            },
                            "error": "weak terminal character support",
                        }
                    ],
                },
            )

            repaired = repair_plan_for_acoustic_safety(
                plan_path=initial_dir / "streaming_plan.json",
                boundary_plan_path=boundary_path,
                output_dir=root / "repair",
                backend=repair_backend,
                retry_index=1,
            )

            self.assertEqual(repaired["status"], "complete")
            self.assertEqual(
                repaired["reconstructed_narration"],
                "And it might begin with the familiar words once upon a time.",
            )
            self.assertFalse(
                any(
                    source_range["start_word_id"] <= 5 < source_range["end_word_id"]
                    for source_range in repaired["selected_source_ranges"]
                )
            )
            self.assertTrue(
                any(
                    source_range["start_word_id"] <= 8 < source_range["end_word_id"]
                    for source_range in repaired["selected_source_ranges"]
                )
            )
            self.assertEqual(
                read_json(root / "repair/grounding_validation.json")["status"],
                "valid",
            )
            self.assertIn("REJECTED ACOUSTIC BOUNDARIES", repair_backend.prompts[0])
            self.assertIn("forbidden_word_ids", repair_backend.prompts[0])
            self.assertEqual(len(repair_backend.prompts), 2)
            self.assertIn("forbidden weak source word 5", repair_backend.prompts[1])
            repair_record = read_json(root / "repair/acoustic_repair.json")["repairs"][
                0
            ]
            self.assertEqual(repair_record["forbidden_word_ids"], [5])
            self.assertEqual(
                repair_record["failure_reasons"], ["weak_terminal_word_support"]
            )

    def test_acoustic_repair_persists_successful_thoughts_when_a_later_one_fails(
        self,
    ) -> None:
        transcript = transcript_with_word_groups(
            [
                (
                    0.0,
                    [
                        "Alpha",
                        "complete.",
                        "Alpha",
                        "complete.",
                        "Beta",
                        "complete.",
                        "Beta",
                        "complete.",
                    ],
                )
            ]
        )
        initial_backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "Alpha complete.",
                            (0, 1, "Alpha", "complete.", "Alpha complete."),
                        )
                    ],
                    pending_start_word_id=4,
                ),
                response(
                    finalized=[
                        thought(
                            "Beta complete.",
                            (4, 5, "Beta", "complete.", "Beta complete."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )
        invalid_second_thought = response(
            finalized=[
                thought(
                    "Beta complete.",
                    (4, 5, "Beta", "complete.", "Beta complete."),
                )
            ],
            pending_start_word_id=None,
        )
        repair_backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "Alpha complete.",
                            (2, 3, "Alpha", "complete.", "Alpha complete."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
                invalid_second_thought,
                invalid_second_thought,
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            initial_dir = root / "initial"
            run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=initial_dir,
                backend=initial_backend,
            )
            boundary_path = root / "rejected_boundary_plan.json"
            write_json(
                boundary_path,
                {
                    "status": "unsafe",
                    "active_repair_boundary_ids": ["alpha_edge", "beta_edge"],
                    "boundaries": [
                        {
                            "boundary_id": "alpha_edge",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "retained_thought_indices": [0],
                            "source_word_ids": {
                                "last_retained_left": 1,
                                "first_omitted": 2,
                            },
                            "forbidden_word_ids": [1],
                            "failure_reason": "weak_terminal_word_support",
                        },
                        {
                            "boundary_id": "beta_edge",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "retained_thought_indices": [1],
                            "source_word_ids": {
                                "last_retained_left": 5,
                                "first_omitted": 6,
                            },
                            "forbidden_word_ids": [5],
                            "failure_reason": "weak_terminal_word_support",
                        },
                    ],
                },
            )

            repaired = repair_plan_for_acoustic_safety(
                plan_path=initial_dir / "streaming_plan.json",
                boundary_plan_path=boundary_path,
                output_dir=root / "repair",
                backend=repair_backend,
                retry_index=1,
            )

            self.assertEqual(
                repaired["committed"][0]["source_ranges"][0]["start_word_id"],
                2,
            )
            self.assertEqual(
                repaired["committed"][1]["source_ranges"][0]["start_word_id"],
                4,
            )
            self.assertTrue(repaired["acoustic_repair_partial"])
            self.assertEqual(len(repaired["acoustic_repairs"]), 1)
            self.assertEqual(len(repaired["acoustic_repair_failures"]), 1)
            self.assertEqual(
                repaired["acoustic_repair_failures"][0]["thought_index"],
                1,
            )
            self.assertEqual(
                read_json(root / "repair/streaming_plan.json"),
                repaired,
            )
            self.assertEqual(
                read_json(root / "repair/grounding_validation.json")["status"],
                "valid",
            )
            ledger = read_json(root / "repair/acoustic_repair.json")
            self.assertTrue(ledger["partial"])
            self.assertEqual(len(ledger["repairs"]), 1)
            self.assertEqual(len(ledger["failures"]), 1)
            self.assertEqual(len(repair_backend.prompts), 3)
            self.assertEqual(list((root / "repair").glob("*.wav")), [])

    def test_acoustic_repair_uses_history_as_a_constraint_not_a_new_target(
        self,
    ) -> None:
        transcript = transcript_with_word_groups(
            [
                (
                    0.0,
                    [
                        "Alpha",
                        "complete.",
                        "unused",
                        "material",
                        "Beta",
                        "complete.",
                        "Beta",
                        "complete.",
                        "Beta",
                        "complete.",
                    ],
                )
            ]
        )
        initial_backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "Alpha complete.",
                            (0, 1, "Alpha", "complete.", "Alpha complete."),
                        )
                    ],
                    pending_start_word_id=2,
                ),
                response(
                    finalized=[
                        thought(
                            "Beta complete.",
                            (6, 7, "Beta", "complete.", "Beta complete."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )
        repair_backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "Beta complete.",
                            (4, 5, "Beta", "complete.", "Beta complete."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
                response(
                    finalized=[
                        thought(
                            "Beta complete.",
                            (8, 9, "Beta", "complete.", "Beta complete."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            initial_dir = root / "initial"
            run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=initial_dir,
                backend=initial_backend,
            )
            boundary_path = root / "rejected_boundary_plan.json"
            write_json(
                boundary_path,
                {
                    "status": "unsafe",
                    "active_repair_boundary_ids": ["current_beta_edge"],
                    "boundaries": [
                        {
                            "boundary_id": "current_beta_edge",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "retained_thought_indices": [1],
                            "source_word_ids": {
                                "last_retained_left": 7,
                                "first_omitted": 8,
                            },
                            "forbidden_word_ids": [7],
                            "failure_reason": "weak_terminal_word_support",
                        },
                        {
                            "boundary_id": "retry_01:old_beta_edge",
                            "repair_constraint_role": "historical",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "retained_thought_indices": [1],
                            "source_word_ids": {
                                "last_retained_left": 5,
                                "first_omitted": 6,
                            },
                            "forbidden_word_ids": [5],
                            "failure_reason": "weak_terminal_word_support",
                        },
                        {
                            "boundary_id": "retry_01:old_alpha_edge",
                            "repair_constraint_role": "historical",
                            "boundary_kind": "selected_to_omitted",
                            "safety_status": "weak_retained_word_alignment",
                            "retained_thought_indices": [0],
                            "source_word_ids": {
                                "last_retained_left": 1,
                                "first_omitted": 2,
                            },
                            "forbidden_word_ids": [1],
                            "failure_reason": "weak_terminal_word_support",
                        },
                    ],
                },
            )

            repaired = repair_plan_for_acoustic_safety(
                plan_path=initial_dir / "streaming_plan.json",
                boundary_plan_path=boundary_path,
                output_dir=root / "repair",
                backend=repair_backend,
                retry_index=2,
            )

            self.assertEqual(
                repaired["committed"][0]["source_ranges"][0]["start_word_id"],
                0,
            )
            self.assertEqual(
                repaired["committed"][1]["source_ranges"][0]["start_word_id"],
                8,
            )
            self.assertEqual(len(repair_backend.prompts), 2)
            self.assertIn("retry_01:old_beta_edge", repair_backend.prompts[0])
            self.assertNotIn("retry_01:old_alpha_edge", repair_backend.prompts[0])
            self.assertIn("forbidden weak source word 5", repair_backend.prompts[1])
            repair_record = read_json(root / "repair/acoustic_repair.json")["repairs"][
                0
            ]
            self.assertEqual(
                repair_record["unsafe_boundary_ids"], ["current_beta_edge"]
            )
            self.assertEqual(
                repair_record["constraint_boundary_ids"],
                ["current_beta_edge", "retry_01:old_beta_edge"],
            )
            self.assertEqual(repair_record["forbidden_word_ids"], [5, 7])

    def test_streaming_delay_replaces_attempt_and_preserves_future_thoughts(
        self,
    ) -> None:
        transcript = transcript_with_word_groups(
            [
                (0.0, ["I", "want", "to", "describe", "uh"]),
                (10.0, ["I", "will", "explain", "the", "algorithm."]),
                (35.0, ["First,", "consider", "a", "weight", "matrix."]),
                (70.0, ["Next,", "score", "every", "parameter", "carefully."]),
            ]
        )
        backend = FakeStreamingBackend(
            [
                response(finalized=[], pending_start_word_id=5),
                response(
                    finalized=[
                        thought(
                            "I will explain the algorithm.",
                            (
                                5,
                                9,
                                "I",
                                "algorithm.",
                                "I will explain the algorithm.",
                            ),
                        )
                    ],
                    pending_start_word_id=10,
                ),
                response(
                    finalized=[
                        thought(
                            "First, consider a weight matrix.",
                            (
                                10,
                                14,
                                "First,",
                                "matrix.",
                                "First, consider a weight matrix.",
                            ),
                        )
                    ],
                    pending_start_word_id=15,
                ),
                response(
                    finalized=[
                        thought(
                            "Next, score every parameter carefully.",
                            (
                                15,
                                19,
                                "Next,",
                                "carefully.",
                                "Next, score every parameter carefully.",
                            ),
                        )
                    ],
                    pending_start_word_id=None,
                    pending_reason="EOF finalized",
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["status"], "complete")
            self.assertEqual(
                plan["reconstructed_narration"],
                (
                    "I will explain the algorithm. "
                    "First, consider a weight matrix. "
                    "Next, score every parameter carefully."
                ),
            )
            self.assertEqual(
                plan["selected_source_ranges"],
                [
                    {"start_word_id": 5, "end_word_id": 10},
                    {"start_word_id": 10, "end_word_id": 15},
                    {"start_word_id": 15, "end_word_id": 20},
                ],
            )
            self.assertEqual(
                [
                    word["id"]
                    for word in plan["iterations"][0]["complete_pending_input"]
                ],
                list(range(10)),
            )
            self.assertEqual(
                [
                    word["id"]
                    for word in plan["iterations"][0]["remaining_pending_transcript"]
                ],
                list(range(5, 10)),
            )
            self.assertEqual(
                [
                    word["id"]
                    for word in plan["iterations"][1]["complete_pending_input"]
                ],
                list(range(5, 15)),
            )
            saved = read_json(output_dir / "streaming_plan.json")
            self.assertEqual(
                saved["reconstructed_narration"], plan["reconstructed_narration"]
            )
            self.assertFalse(any(output_dir.glob("*.wav")))

    def test_unique_prefix_may_use_two_ordered_source_ranges(self) -> None:
        words = [
            TranscriptWord(index, text, float(index), index + 0.2)
            for index, text in enumerate(
                [
                    "In",
                    "this",
                    "video",
                    "I",
                    "started",
                    "wrong",
                    "I",
                    "explain",
                    "the",
                    "model.",
                    "Next",
                    "thought.",
                ]
            )
        ]
        raw = response(
            finalized=[
                thought(
                    "In this video I explain the model.",
                    (0, 2, "In", "video", "In this video"),
                    (6, 9, "I", "model.", "I explain the model."),
                )
            ],
            pending_start_word_id=10,
        )

        decision = validate_decision(
            raw,
            pending_words=words,
            final_pass=False,
            committed_source_end=0,
        )

        self.assertEqual(
            decision.finalized[0].source_ranges[1].start_word_id,
            6,
        )

    def test_invalid_first_response_is_retried_with_validation_error(self) -> None:
        transcript = transcript_with_word_groups(
            [(0.0, ["A", "complete", "thought.", "Next"])]
        )
        backend = FakeStreamingBackend(
            [
                response(
                    finalized=[
                        thought(
                            "Invented",
                            (999, 999, "Invented", "Invented", "Invented"),
                        )
                    ],
                    pending_start_word_id=0,
                ),
                response(finalized=[], pending_start_word_id=0),
                response(
                    finalized=[
                        thought(
                            "A complete thought. Next",
                            (
                                0,
                                3,
                                "A",
                                "Next",
                                "A complete thought. Next",
                            ),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["status"], "complete")
            self.assertIn("VALIDATION ERROR", backend.prompts[1])
            self.assertIn("999", backend.prompts[1])
            self.assertTrue((output_dir / "iteration_0001_attempt_1.raw.txt").exists())
            self.assertTrue((output_dir / "iteration_0001_attempt_2.raw.json").exists())

    def test_malformed_window_fails_soft_with_exact_source_passthrough(self) -> None:
        transcript = transcript_with_word_groups([(0.0, ["One", "thought."])])
        backend = FakeStreamingBackend(
            ["not json", "still not json", "bad eof", "bad eof again"]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            failure = read_json(output_dir / "iteration_0001_failure.json")
            self.assertEqual(len(failure["attempts"]), 2)
            self.assertEqual(plan["status"], "complete")
            self.assertEqual(
                plan["reconstructed_narration"],
                "One thought.",
            )
            self.assertEqual(
                plan["selected_source_ranges"],
                [{"start_word_id": 0, "end_word_id": 2}],
            )
            self.assertEqual(plan["fallback_status"], "source_passthrough_used")
            self.assertEqual(
                [item["status"] for item in plan["fallbacks"]],
                [
                    "source_passthrough_deferred_for_lookahead",
                    "eof_source_passthrough",
                ],
            )
            self.assertEqual(
                [item["trigger"] for item in plan["fallbacks"]],
                ["request_failure", "request_failure"],
            )
            self.assertTrue(
                all(
                    not item["rejected_model_output_accepted"]
                    for item in plan["fallbacks"]
                )
            )
            grounding = read_json(output_dir / "grounding_validation.json")
            self.assertEqual(grounding["status"], "valid")
            self.assertTrue(grounding["plan_accepted"])
            self.assertEqual(grounding["fallback_count"], 2)

    def test_deferred_failure_that_later_resolves_is_not_counted_as_passthrough(
        self,
    ) -> None:
        transcript = transcript_with_word_groups(
            [(0.0, ["First", "thought."]), (35.0, ["Second", "thought."])]
        )
        backend = FakeStreamingBackend(
            [
                "bad response",
                "still bad",
                response(
                    finalized=[
                        thought(
                            "First thought.",
                            (0, 1, "First", "thought.", "First thought."),
                        )
                    ],
                    pending_start_word_id=2,
                ),
                response(
                    finalized=[
                        thought(
                            "Second thought.",
                            (2, 3, "Second", "thought.", "Second thought."),
                        )
                    ],
                    pending_start_word_id=None,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["committed_words"], [0, 1, 2, 3])
            self.assertEqual(len(plan["fallbacks"]), 1)
            self.assertEqual(plan["fallbacks"][0]["source_ranges"], [])
            self.assertEqual(
                plan["fallback_status"],
                "planner_failure_recovered_without_source_passthrough",
            )
            self.assertEqual(plan["fallback_event_count"], 1)
            self.assertEqual(plan["source_passthrough_count"], 0)
            grounding = read_json(output_dir / "grounding_validation.json")
            self.assertEqual(grounding["fallback_count"], 1)
            self.assertEqual(grounding["source_passthrough_count"], 0)

    def test_unsupported_response_is_rejected_then_source_is_preserved(self) -> None:
        transcript = transcript_with_word_groups(
            [
                (
                    0.0,
                    ["This", "is", "a", "very", "simple", "example", "Next"],
                )
            ]
        )
        invalid = response(
            finalized=[
                thought(
                    "This is a very simple example",
                    (
                        0,
                        4,
                        "This",
                        "simple",
                        "This is a very simple example",
                    ),
                )
            ],
            pending_start_word_id=5,
        )
        backend = FakeStreamingBackend([invalid, invalid, invalid, invalid])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            grounding = read_json(output_dir / "grounding_validation.json")
            self.assertEqual(plan["status"], "complete")
            self.assertEqual(
                plan["selected_source_ranges"],
                [{"start_word_id": 0, "end_word_id": 7}],
            )
            self.assertEqual(grounding["status"], "valid")
            self.assertTrue(grounding["plan_accepted"])
            self.assertEqual(grounding["planner_retries"], 2)
            self.assertEqual(grounding["unsupported_tokens"], [])
            self.assertEqual(
                grounding["fallbacks"][0]["grounding_failure"][
                    "unsupported_canonical_token"
                ],
                "example",
            )
            self.assertEqual(
                grounding["fallbacks"][1]["grounding_failure"][
                    "unsupported_canonical_token"
                ],
                "example",
            )

    def test_final_eof_failure_preserves_every_pending_source_word(self) -> None:
        transcript = transcript_with_word_groups(
            [(0.0, ["Keep", "every", "pending", "word."])]
        )
        backend = FakeStreamingBackend(
            [
                response(finalized=[], pending_start_word_id=0),
                "malformed eof",
                "still malformed eof",
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(
                plan["reconstructed_narration"], "Keep every pending word."
            )
            self.assertEqual(plan["committed_words"], [0, 1, 2, 3])
            self.assertEqual(len(plan["fallbacks"]), 1)
            fallback = plan["fallbacks"][0]
            self.assertEqual(fallback["status"], "eof_source_passthrough")
            self.assertEqual(fallback["trigger"], "request_failure")
            self.assertEqual(
                fallback["source_ranges"],
                [{"start_word_id": 0, "end_word_id": 4}],
            )
            self.assertEqual(fallback["passthrough_word_ids"], [0, 1, 2, 3])
            self.assertIsNone(fallback["remaining_pending_range"])
            validation = plan["committed"][0]["grounding_validation"]
            self.assertEqual(
                validation["grounding_mode"],
                "deterministic_exact_source_passthrough",
            )

    def test_repeated_window_failures_continue_chronologically(self) -> None:
        transcript = transcript_with_word_groups(
            [
                (0.0, ["Window", "zero."]),
                (35.0, ["Window", "one."]),
                (70.0, ["Window", "two."]),
                (105.0, ["Window", "three."]),
            ]
        )
        backend = FakeStreamingBackend(["bad response"] * 10)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["status"], "complete")
            self.assertEqual(plan["committed_words"], list(range(8)))
            self.assertEqual(
                plan["selected_source_ranges"],
                [
                    {"start_word_id": 0, "end_word_id": 2},
                    {"start_word_id": 2, "end_word_id": 4},
                    {"start_word_id": 4, "end_word_id": 6},
                    {"start_word_id": 6, "end_word_id": 8},
                ],
            )
            self.assertEqual(len(plan["iterations"]), 5)
            self.assertEqual(len(plan["fallbacks"]), 5)
            self.assertEqual(
                plan["iterations"][1]["remaining_pending_transcript"][0]["id"],
                2,
            )
            self.assertTrue(
                all(
                    right["start_word_id"] == left["end_word_id"]
                    for left, right in zip(
                        plan["selected_source_ranges"],
                        plan["selected_source_ranges"][1:],
                        strict=False,
                    )
                )
            )

    def test_repeated_valid_no_progress_waits_for_later_successful_take(self) -> None:
        transcript = transcript_with_word_groups(
            [
                (0.0, ["I", "want", "to", "explain", "text."]),
                (35.0, ["I", "want", "to", "explain", "the", "audio."]),
                (70.0, ["Next", "topic", "starts."]),
            ]
        )
        backend = FakeStreamingBackend(
            [
                response(finalized=[], pending_start_word_id=0),
                response(finalized=[], pending_start_word_id=0),
                response(
                    finalized=[
                        thought(
                            "I want to explain the audio.",
                            (
                                5,
                                10,
                                "I",
                                "audio.",
                                "I want to explain the audio.",
                            ),
                        )
                    ],
                    pending_start_word_id=11,
                ),
                response(finalized=[], pending_start_word_id=None),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["status"], "complete")
            self.assertEqual(
                plan["reconstructed_narration"], "I want to explain the audio."
            )
            self.assertEqual(plan["committed_words"], list(range(5, 11)))
            self.assertEqual(
                plan["selected_source_ranges"],
                [{"start_word_id": 5, "end_word_id": 11}],
            )
            self.assertEqual(plan["fallbacks"], [])
            self.assertEqual(plan["source_passthrough_count"], 0)
            self.assertEqual(
                [
                    len(iteration["complete_pending_input"])
                    for iteration in plan["iterations"][:-1]
                ],
                [5, 11, 14],
            )
            self.assertEqual(plan["iterations"][0]["pending_no_progress_age"], 1)
            self.assertEqual(plan["iterations"][1]["pending_no_progress_age"], 2)
            self.assertIsNone(plan["iterations"][0]["fallback"])
            self.assertIsNone(plan["iterations"][1]["fallback"])
            self.assertNotIn(0, plan["committed_words"])
            self.assertNotIn(4, plan["committed_words"])
            self.assertTrue(
                all(len(iteration["attempts"]) == 1 for iteration in plan["iterations"])
            )

    def test_valid_empty_eof_response_discards_abandoned_pending_source(self) -> None:
        transcript = transcript_with_word_groups(
            [(0.0, ["Do", "not", "drop", "this."])]
        )
        backend = FakeStreamingBackend(
            [
                response(finalized=[], pending_start_word_id=0),
                response(finalized=[], pending_start_word_id=None),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            self.assertEqual(plan["reconstructed_narration"], "")
            self.assertEqual(plan["committed_words"], [])
            self.assertEqual(plan["selected_source_ranges"], [])
            self.assertEqual(plan["fallbacks"], [])
            self.assertEqual(plan["fallback_status"], "not_used")
            self.assertEqual(plan["source_passthrough_count"], 0)
            self.assertNotIn(
                "iteration_0002_failure.json", {p.name for p in output_dir.iterdir()}
            )
            grounding = read_json(output_dir / "grounding_validation.json")
            self.assertEqual(grounding["status"], "valid")
            self.assertEqual(grounding["fallbacks"], [])
            self.assertEqual(grounding["source_passthrough_count"], 0)

    def test_validation_rejects_unsafe_range_geometry(self) -> None:
        words = [
            TranscriptWord(index, str(index), float(index), index + 0.2)
            for index in range(8)
        ]
        invalid_responses = [
            response(
                finalized=[
                    thought(
                        "unknown",
                        (8, 8, "unknown", "unknown", "unknown"),
                    )
                ],
                pending_start_word_id=7,
            ),
            response(
                finalized=[
                    thought(
                        "zero one two three three four",
                        (
                            0,
                            3,
                            "0",
                            "3",
                            "zero one two three",
                        ),
                        (3, 4, "3", "4", "three four"),
                    )
                ],
                pending_start_word_id=5,
            ),
            response(
                finalized=[
                    thought(
                        "zero one two three four five",
                        (
                            0,
                            5,
                            "0",
                            "5",
                            "zero one two three four five",
                        ),
                    )
                ],
                pending_start_word_id=5,
            ),
            response(
                finalized=[
                    {
                        "canonical_text": "no source",
                        "source_ranges": [],
                    }
                ],
                pending_start_word_id=5,
            ),
        ]

        for raw in invalid_responses:
            with self.subTest(raw=raw):
                with self.assertRaises(DecisionValidationError):
                    validate_decision(
                        raw,
                        pending_words=words,
                        final_pass=False,
                        committed_source_end=0,
                    )

    def test_loader_uses_words_not_chunk_boundaries(self) -> None:
        transcript = {
            "chunks": [{"text": "wrong display-only words", "timestamp": [0, 9]}],
            "atoms": [
                {
                    "atom_index": 1,
                    "words": [{"word": "third", "start": 2.0, "end": 2.2}],
                },
                {
                    "atom_index": 0,
                    "words": [
                        {"word": "first", "start": 0.0, "end": 0.2},
                        {"word": "second", "start": 1.0, "end": 1.2},
                    ],
                },
            ],
        }

        words = load_transcript_words(transcript)

        self.assertEqual([word.id for word in words], [0, 1, 2])
        self.assertEqual(
            [word.text for word in words],
            ["first", "second", "third"],
        )

    def test_gemini_schema_uses_inclusive_anchored_ranges(self) -> None:
        source_range = streaming_response_schema()["properties"]["finalized"]["items"][
            "properties"
        ]["source_ranges"]["items"]

        self.assertEqual(
            set(source_range["required"]),
            {
                "first_word_id",
                "last_word_id",
                "first_word",
                "last_word",
                "canonical_text",
            },
        )
        self.assertNotIn("start_word_id", source_range["properties"])
        self.assertNotIn("end_word_id", source_range["properties"])
        self.assertIn(
            "Inclusive",
            source_range["properties"]["last_word_id"]["description"],
        )

    def test_source_grounding_rejects_off_by_one_example_range(self) -> None:
        texts = [
            "This",
            "is",
            "a",
            "very",
            "simple",
            "example",
            "how",
        ]
        words = [
            TranscriptWord(11 + index, text, index * 0.5, index * 0.5 + 0.4)
            for index, text in enumerate(texts)
        ]
        invalid = response(
            finalized=[
                thought(
                    "This is a very simple example",
                    (
                        11,
                        15,
                        "This",
                        "simple",
                        "This is a very simple example",
                    ),
                )
            ],
            pending_start_word_id=16,
        )

        with self.assertRaises(SourceGroundingValidationError) as caught:
            validate_decision(
                invalid,
                pending_words=words,
                final_pass=False,
                committed_source_end=11,
            )

        message = str(caught.exception)
        self.assertIn('Unsupported canonical token "example"', message)
        self.assertIn('range [11, 16) ends at source word "simple"', message)
        self.assertIn('source word 16 "example" is outside', message)
        self.assertIn("word 15: simple", message)
        self.assertIn("word 16: example", message)
        self.assertIn("word 17: how", message)

    def test_grounding_allows_supported_asr_normalizations(self) -> None:
        source_texts = [
            "row",
            "Master",
            "it",
            "wave",
            "Abundant",
            "closes",
            "examples",
            "Next",
        ]
        words = [
            TranscriptWord(index, text, float(index), index + 0.2)
            for index, text in enumerate(source_texts)
        ]
        raw = response(
            finalized=[
                thought(
                    "raw mastered WAV abandoned clauses example",
                    (0, 0, "row", "row", "raw"),
                    (
                        1,
                        3,
                        "Master",
                        "wave",
                        "mastered WAV",
                    ),
                    (
                        4,
                        5,
                        "Abundant",
                        "closes",
                        "abandoned clauses",
                    ),
                    (6, 6, "examples", "examples", "example"),
                )
            ],
            pending_start_word_id=7,
        )

        decision = validate_decision(
            raw,
            pending_words=words,
            final_pass=False,
            committed_source_end=0,
        )

        validation = decision.finalized[0].grounding_validation
        self.assertEqual(validation["unsupported_tokens"], [])
        self.assertEqual(
            validation["canonical_tokens"],
            validation["supported_tokens"],
        )

    def test_grounding_reports_a_skipped_range_prefix_instead_of_plural_tail(
        self,
    ) -> None:
        words = [
            TranscriptWord(index, text, float(index), index + 0.2)
            for index, text in enumerate(
                ["more.", "ask", "complimentary", "questions."]
            )
        ]
        invalid = response(
            finalized=[
                thought(
                    "ask complementary questions.",
                    (
                        0,
                        3,
                        "more.",
                        "questions.",
                        "ask complementary questions.",
                    ),
                )
            ],
            pending_start_word_id=None,
        )

        with self.assertRaises(SourceGroundingValidationError) as caught:
            validate_decision(
                invalid,
                pending_words=words,
                final_pass=True,
                committed_source_end=0,
            )

        self.assertIn('Unsupported canonical token "ask"', str(caught.exception))
        self.assertEqual(
            caught.exception.report["unsupported_canonical_token"],
            "ask",
        )

    def test_grounding_rejects_unrepresented_audio_inside_selected_range(
        self,
    ) -> None:
        words = [
            TranscriptWord(index, text, float(index), index + 0.2)
            for index, text in enumerate(["This", "is", "um", "a", "test", "Next"])
        ]
        hidden_filler = response(
            finalized=[
                thought(
                    "This is a test",
                    (0, 4, "This", "test", "This is a test"),
                )
            ],
            pending_start_word_id=5,
        )

        with self.assertRaises(SourceGroundingValidationError) as caught:
            validate_decision(
                hidden_filler,
                pending_words=words,
                final_pass=False,
                committed_source_end=0,
            )
        self.assertIn(
            'Unrepresented selected source token "um"',
            str(caught.exception),
        )
        self.assertEqual(
            caught.exception.report["unrepresented_source_tokens"],
            [
                {
                    "source_token_index": 2,
                    "source_word_id": 2,
                    "source_token": "um",
                }
            ],
        )

        split_around_filler = response(
            finalized=[
                thought(
                    "This is a test",
                    (0, 1, "This", "is", "This is"),
                    (3, 4, "a", "test", "a test"),
                )
            ],
            pending_start_word_id=5,
        )
        accepted = validate_decision(
            split_around_filler,
            pending_words=words,
            final_pass=False,
            committed_source_end=0,
        )
        self.assertEqual(
            [
                (item.start_word_id, item.end_word_id)
                for item in accepted.finalized[0].source_ranges
            ],
            [(0, 2), (3, 5)],
        )

    def test_incremental_commit_rejects_take_repeated_by_pending_suffix(
        self,
    ) -> None:
        texts = [
            "We",
            "explain",
            "how",
            "the",
            "pruning",
            "model",
            "works.",
            "how",
            "the",
            "pruning",
            "model",
            "works",
            "in",
            "practice.",
        ]
        words = [
            TranscriptWord(index, text, float(index), index + 0.2)
            for index, text in enumerate(texts)
        ]
        finalized_too_soon = response(
            finalized=[
                thought(
                    "We explain how the pruning model works.",
                    (
                        0,
                        6,
                        "We",
                        "works.",
                        "We explain how the pruning model works.",
                    ),
                )
            ],
            pending_start_word_id=7,
        )

        with self.assertRaisesRegex(
            DecisionValidationError,
            "pending suffix repeats finalized phrase",
        ):
            validate_decision(
                finalized_too_soon,
                pending_words=words,
                final_pass=False,
                committed_source_end=0,
            )

    def test_thought_text_must_equal_range_text_concatenation(self) -> None:
        words = [
            TranscriptWord(0, "One", 0.0, 0.2),
            TranscriptWord(1, "thought", 0.3, 0.5),
            TranscriptWord(2, "Next", 0.6, 0.8),
        ]
        raw = response(
            finalized=[
                thought(
                    "One thought invented",
                    (0, 1, "One", "thought", "One thought"),
                )
            ],
            pending_start_word_id=2,
        )

        with self.assertRaisesRegex(
            DecisionValidationError,
            "does not equal the concatenation",
        ):
            validate_decision(
                raw,
                pending_words=words,
                final_pass=False,
                committed_source_end=0,
            )

    def test_off_by_one_is_retried_and_never_reaches_accepted_plan(
        self,
    ) -> None:
        transcript = transcript_with_word_groups(
            [
                (
                    0.0,
                    [
                        "discard",
                        "these",
                        "earlier",
                        "abandoned",
                        "attempt",
                        "words",
                        "before",
                        "the",
                        "successful",
                        "final",
                        "take",
                        "This",
                        "is",
                        "a",
                        "very",
                        "simple",
                        "example",
                        "Next",
                    ],
                )
            ]
        )
        invalid = response(
            finalized=[
                thought(
                    "This is a very simple example",
                    (
                        11,
                        15,
                        "This",
                        "simple",
                        "This is a very simple example",
                    ),
                )
            ],
            pending_start_word_id=16,
        )
        corrected = response(
            finalized=[
                thought(
                    "This is a very simple example",
                    (
                        11,
                        16,
                        "This",
                        "example",
                        "This is a very simple example",
                    ),
                )
            ],
            pending_start_word_id=17,
        )
        final = response(
            finalized=[
                thought(
                    "Next",
                    (17, 17, "Next", "Next", "Next"),
                )
            ],
            pending_start_word_id=None,
            pending_reason="EOF finalized",
        )
        backend = FakeStreamingBackend([invalid, corrected, final])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_path = root / "source_transcript.json"
            write_json(transcript_path, transcript)
            output_dir = root / "plan"

            plan = run_streaming_planner(
                transcript_path=transcript_path,
                output_dir=output_dir,
                backend=backend,
            )

            first_range = plan["committed"][0]["source_ranges"][0]
            self.assertEqual(
                {
                    "start_word_id": first_range["start_word_id"],
                    "end_word_id": first_range["end_word_id"],
                },
                {"start_word_id": 11, "end_word_id": 17},
            )
            self.assertEqual(first_range["last_word_id"], 16)
            self.assertEqual(first_range["last_word"], "example")
            self.assertNotEqual(
                (
                    plan["words"][first_range["end_word_id"] - 1]["text"],
                    plan["words"][first_range["end_word_id"]]["text"],
                ),
                ("simple", "example"),
            )
            self.assertIn('Unsupported canonical token "example"', backend.prompts[1])
            self.assertIn("word 15: simple", backend.prompts[1])
            self.assertIn("word 16: example", backend.prompts[1])
            self.assertIn("word 17: Next", backend.prompts[1])

            grounding = read_json(output_dir / "grounding_validation.json")
            self.assertEqual(grounding["status"], "valid")
            self.assertEqual(grounding["planner_retries"], 1)
            self.assertEqual(grounding["unsupported_tokens"], [])
            self.assertTrue(grounding["plan_accepted"])
            first_validation = grounding["thoughts"][0]
            self.assertEqual(first_validation["thought_index"], 0)
            self.assertEqual(first_validation["canonical_tokens"], 6)
            self.assertEqual(first_validation["supported_tokens"], 6)
            self.assertEqual(first_validation["unsupported_tokens"], [])
            self.assertEqual(first_validation["status"], "valid")


if __name__ == "__main__":
    unittest.main()
