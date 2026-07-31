#!/usr/bin/env python3
"""Fail-closed final rendering for an existing grounded semantic plan."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

from .common import read_json, sha256_file, write_json
from .hard_align import (
    DEFAULT_ALIGNMENT_PYTHON,
    render_forced_aligned_preview,
)
from .leading_align import render_leading_aligned_preview
from .planner_backends import (
    DEFAULT_LOCAL_PYTHON,
    DEFAULT_MAX_OUTPUT_TOKENS,
    PAUSE_SYSTEM_INSTRUCTION,
    PlannerBackend as PausePlannerBackend,
    add_planner_backend_arguments,
    create_planner_backend,
)
from .rough_render import (
    flatten_selected_ranges,
    load_plan_words,
)
from .semantic_pause import (
    create_pause_plan,
    render_semantic_pauses,
)
from .trailing_refine import render_trailing_refined_preview


class FinalRenderError(RuntimeError):
    """A safe final cut cannot be published."""


def _validate_grounded_plan(
    *,
    audio_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], Path, int]:
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("status") != "complete":
        raise FinalRenderError("final rendering requires a complete streaming plan")
    if plan.get("planner") != "streaming_narration_v1":
        raise FinalRenderError(
            "final rendering requires a current streaming narration plan"
        )
    words = load_plan_words(plan)
    ranges = flatten_selected_ranges(plan, word_count=len(words))
    if not ranges:
        raise FinalRenderError("streaming plan selects no source ranges")
    committed = plan.get("committed")
    if not isinstance(committed, list) or not committed:
        raise FinalRenderError("streaming plan contains no committed thoughts")

    transcript_value = plan.get("transcript")
    if not isinstance(transcript_value, str) or not transcript_value:
        raise FinalRenderError(
            "streaming plan has no immutable source-transcript provenance"
        )
    transcript_path = Path(transcript_value).resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    transcript = read_json(transcript_path)
    if not isinstance(transcript, dict):
        raise FinalRenderError("source transcript root must be an object")
    expected_audio_sha = transcript.get("audio_sha256")
    if not isinstance(expected_audio_sha, str):
        raise FinalRenderError("source transcript has no audio SHA-256")
    if sha256_file(audio_path) != expected_audio_sha:
        raise FinalRenderError(
            "audio does not match the recording used by the semantic plan"
        )
    expected_transcript_sha = plan.get("transcript_sha256")
    if (
        isinstance(expected_transcript_sha, str)
        and sha256_file(transcript_path) != expected_transcript_sha
    ):
        raise FinalRenderError("source transcript changed after semantic planning")

    grounding_value = plan.get("grounding_validation")
    if not isinstance(grounding_value, str) or not grounding_value:
        raise FinalRenderError(
            "final rendering requires a strict source-grounding validation"
        )
    grounding_path = Path(grounding_value).resolve()
    if not grounding_path.is_file():
        raise FileNotFoundError(grounding_path)
    grounding = read_json(grounding_path)
    if not isinstance(grounding, dict):
        raise FinalRenderError("grounding validation root must be an object")
    if grounding.get("validator") != "strict_bidirectional_range_source_grounding_v2":
        raise FinalRenderError(
            "grounding report was not produced by the strict bidirectional "
            "source-range validator"
        )
    if (
        grounding.get("status") != "valid"
        or grounding.get("plan_accepted") is not True
        or grounding.get("unsupported_tokens") != []
        or grounding.get("unrepresented_source_tokens") != []
    ):
        raise FinalRenderError(
            "semantic plan did not pass strict source-grounding validation"
        )
    canonical_tokens = grounding.get("canonical_tokens")
    supported_tokens = grounding.get("supported_tokens")
    if (
        type(canonical_tokens) is not int
        or type(supported_tokens) is not int
        or canonical_tokens < 1
        or supported_tokens != canonical_tokens
    ):
        raise FinalRenderError("grounding validation has unsupported canonical text")
    if grounding.get("finalized_thoughts") != len(committed):
        raise FinalRenderError(
            "grounding validation does not describe every committed thought"
        )
    if grounding.get("source_ranges") != len(ranges):
        raise FinalRenderError(
            "grounding validation does not describe every selected source range"
        )

    grounding_thoughts = grounding.get("thoughts")
    if not isinstance(grounding_thoughts, list) or len(grounding_thoughts) != len(
        committed
    ):
        raise FinalRenderError(
            "grounding validation thought ledger does not match the semantic plan"
        )
    for thought_index, (thought, validation) in enumerate(
        zip(committed, grounding_thoughts, strict=True)
    ):
        if not isinstance(thought, dict) or not isinstance(validation, dict):
            raise FinalRenderError(
                f"grounding validation thought {thought_index} is malformed"
            )
        embedded = thought.get("grounding_validation")
        if not isinstance(embedded, dict):
            raise FinalRenderError(
                f"committed thought {thought_index} has no embedded grounding "
                "validation"
            )
        expected = json.loads(json.dumps(embedded))
        expected["thought_index"] = thought_index
        if validation != expected:
            raise FinalRenderError(
                f"grounding validation thought {thought_index} does not match "
                "the committed semantic plan"
            )

    return plan, grounding_path, len(ranges)


def _cache_pause_plan(
    *,
    plan_path: Path,
    output_dir: Path,
    supplied_pause_plan_path: Path | None,
    backend: PausePlannerBackend | None,
    env_file: Path,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    local_python: Path,
    local_files_only: bool,
    max_output_tokens: int,
) -> Path:
    destination = output_dir / "pause_plan.json"
    if supplied_pause_plan_path is not None:
        supplied = supplied_pause_plan_path.resolve()
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        if supplied != destination:
            shutil.copy2(supplied, destination)
        return destination

    owns_backend = backend is None
    active_backend = (
        create_planner_backend(
            provider=provider,
            model=model,
            env_file=env_file.resolve(),
            max_output_tokens=max_output_tokens,
            system_instruction=PAUSE_SYSTEM_INSTRUCTION,
            base_url=base_url,
            api_key_env=api_key_env,
            local_python=local_python.absolute(),
            local_files_only=local_files_only,
        )
        if backend is None
        else backend
    )
    try:
        create_pause_plan(
            plan_path=plan_path,
            output_dir=output_dir,
            backend=active_backend,
        )
    finally:
        if owns_backend:
            active_backend.close()
    return destination


def render_final_cut(
    *,
    audio_path: Path,
    plan_path: Path,
    output_dir: Path,
    pause_plan_path: Path | None = None,
    alignment_python: Path = DEFAULT_ALIGNMENT_PYTHON,
    env_file: Path = Path(".env"),
    planner_backend: str = "gemini",
    planner_model: str | None = None,
    planner_base_url: str | None = None,
    planner_api_key_env: str | None = None,
    planner_python: Path = DEFAULT_LOCAL_PYTHON,
    local_files_only: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    pause_backend: PausePlannerBackend | None = None,
    hard_alignment_payload: dict[str, Any] | None = None,
    leading_alignment_payload: dict[str, Any] | None = None,
    write_debug_artifacts: bool = False,
) -> dict[str, Any]:
    """Render one publication candidate without re-running ASR or semantics."""

    audio_path = audio_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty for final rendering: {output_dir}"
        )
    plan, grounding_path, selected_range_count = _validate_grounded_plan(
        audio_path=audio_path,
        plan_path=plan_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    work = output_dir / "work"
    trailing_dir = work / "01_trailing"
    hard_dir = work / "02_hard_boundaries"
    leading_dir = work / "03_leading_boundaries"
    semantic_dir = work / "04_semantic_pauses"
    trailing = render_trailing_refined_preview(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=trailing_dir,
        write_debug_artifacts=write_debug_artifacts,
    )
    hard = render_forced_aligned_preview(
        refined_manifest_path=(trailing_dir / "render_manifest_refined.json"),
        output_dir=hard_dir,
        alignment_python=alignment_python,
        alignment_payload=hard_alignment_payload,
        write_debug_artifacts=write_debug_artifacts,
    )
    if int(hard.get("alignment_failures", 0)):
        raise FinalRenderError(
            "unresolved trailing forced-alignment boundary; refusing to "
            "publish a potentially clipped word"
        )
    leading = render_leading_aligned_preview(
        aligned_manifest_path=(hard_dir / "render_manifest_forced_aligned.json"),
        output_dir=leading_dir,
        alignment_python=alignment_python,
        alignment_payload=leading_alignment_payload,
        write_debug_artifacts=write_debug_artifacts,
    )
    if (
        int(leading.get("leading_alignment_failures", 0))
        or leading.get("all_leading_boundaries_resolved") is not True
    ):
        raise FinalRenderError(
            "unresolved leading forced-alignment boundary; refusing to "
            "publish a potentially clipped word onset"
        )

    cached_pause_plan = _cache_pause_plan(
        plan_path=plan_path,
        output_dir=output_dir,
        supplied_pause_plan_path=pause_plan_path,
        backend=pause_backend,
        env_file=env_file,
        provider=planner_backend,
        model=planner_model,
        base_url=planner_base_url,
        api_key_env=planner_api_key_env,
        local_python=planner_python,
        local_files_only=local_files_only,
        max_output_tokens=max_output_tokens,
    )
    semantic = render_semantic_pauses(
        render_manifest_path=(
            leading_dir / "render_manifest_full_boundary_aligned.json"
        ),
        pause_plan_path=cached_pause_plan,
        output_dir=semantic_dir,
        write_debug_artifacts=write_debug_artifacts,
    )
    semantic_manifest_path = (semantic_dir / "pause_render_manifest.json").resolve()
    if not semantic_manifest_path.is_file():
        raise FinalRenderError("semantic-pause renderer produced no manifest")
    semantic_wav = Path(str(semantic["rough_cut_with_semantic_pauses_wav"])).resolve()
    pause_plan = read_json(cached_pause_plan)
    if not isinstance(pause_plan, dict):
        raise FinalRenderError("pause plan root must be an object")
    final_path = output_dir / "final_cut.wav"
    shutil.copy2(semantic_wav, final_path)
    final_info = sf.info(final_path)

    manifest = {
        "schema_version": 1,
        "renderer": "streaming_plan_final_render_v1",
        "status": "complete",
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "grounding_validation": str(grounding_path),
        "grounding_validation_sha256": sha256_file(grounding_path),
        "pause_plan": str(cached_pause_plan),
        "pause_plan_sha256": sha256_file(cached_pause_plan),
        "pause_planner_backend": pause_plan.get("backend"),
        "pause_planner_model": pause_plan.get("model"),
        "trailing_manifest": str(
            (trailing_dir / "render_manifest_refined.json").resolve()
        ),
        "hard_boundary_manifest": str(
            (hard_dir / "render_manifest_forced_aligned.json").resolve()
        ),
        "leading_boundary_manifest": str(
            (leading_dir / "render_manifest_full_boundary_aligned.json").resolve()
        ),
        "semantic_pause_manifest": str(semantic_manifest_path),
        "semantic_pause_manifest_sha256": sha256_file(semantic_manifest_path),
        "final_cut_wav": str(final_path.resolve()),
        "final_cut_wav_sha256": sha256_file(final_path),
        "sample_rate": int(final_info.samplerate),
        "channel_count": int(final_info.channels),
        "frame_count": int(final_info.frames),
        "duration_seconds": float(final_info.duration),
        "semantic_thoughts": len(plan["committed"]),
        "selected_source_ranges": selected_range_count,
        "rendered_clips": len(leading["clips"]),
        "debug_artifacts_written": write_debug_artifacts,
        "stable_silence_boundaries": int(trailing["number_of_refined_boundaries"]),
        "hard_boundaries": int(hard["hard_boundaries_found"]),
        "hard_boundaries_aligned": int(hard["successfully_aligned"]),
        "leading_boundaries": int(leading["leading_boundaries_found"]),
        "leading_boundaries_resolved": int(
            leading["leading_boundaries_successfully_aligned"]
        ),
        "leading_boundaries_aligned": int(
            leading["leading_boundaries_successfully_aligned"]
        ),
        "leading_waveform_boundaries": int(
            leading.get("leading_waveform_silence_boundaries", 0)
        ),
        "leading_forced_alignment_boundaries": int(
            leading.get("leading_forced_alignment_boundaries", 0)
        ),
        "unresolved_boundaries": (
            int(hard["alignment_failures"]) + int(leading["leading_alignment_failures"])
        ),
        "clip_joins": semantic["clip_joins"],
        "final_boundary": semantic["final_boundary"],
    }
    write_json(output_dir / "final_render_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicecut --render-plan",
        description=(
            "Render a complete, source-grounded streaming plan with waveform "
            "decay refinement, local WhisperX boundary alignment, semantic "
            "target-total pauses, native source room tone, and a safe EOF tail."
        ),
    )
    parser.add_argument("--render-plan", action="store_true")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pause-plan", type=Path)
    parser.add_argument(
        "--alignment-python",
        type=Path,
        default=DEFAULT_ALIGNMENT_PYTHON,
    )
    add_planner_backend_arguments(parser)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Save per-clip WAVs and diagnostic plots for renderer inspection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.render_plan:
        raise SystemExit("final plan rendering requires --render-plan")
    manifest = render_final_cut(
        audio_path=args.audio,
        plan_path=args.plan,
        output_dir=args.output_dir,
        pause_plan_path=args.pause_plan,
        alignment_python=args.alignment_python,
        env_file=args.env_file,
        planner_backend=args.planner_backend,
        planner_model=args.planner_model,
        planner_base_url=args.planner_base_url,
        planner_api_key_env=args.planner_api_key_env,
        planner_python=args.planner_python,
        local_files_only=args.local_files_only,
        max_output_tokens=args.max_output_tokens,
        write_debug_artifacts=args.debug_artifacts,
    )
    print("\nFINAL CUT CREATED")
    print(f"semantic thoughts: {manifest['semantic_thoughts']}")
    print(f"rendered clips: {manifest['rendered_clips']}")
    print(f"stable-silence boundaries: {manifest['stable_silence_boundaries']}")
    print(f"forced-aligned trailing boundaries: {manifest['hard_boundaries_aligned']}")
    print(
        "waveform-resolved leading boundaries: "
        f"{manifest['leading_waveform_boundaries']}"
    )
    print(
        "forced-aligned dense leading boundaries: "
        f"{manifest['leading_forced_alignment_boundaries']}"
    )
    print(f"unresolved boundaries: {manifest['unresolved_boundaries']}")
    print(f"duration: {manifest['duration_seconds']:.3f} s")
    print(f"output path: {manifest['final_cut_wav']}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "final_cut": manifest["final_cut_wav"],
                "manifest": str(
                    (args.output_dir.resolve() / "final_render_manifest.json")
                ),
                "cached_pause_plan": manifest["pause_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
