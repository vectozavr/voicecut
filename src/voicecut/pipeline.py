#!/usr/bin/env python3
"""One-command raw narration + script -> verified production WAV pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .common import parse_script, probe_audio, sha256_file, write_json


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PACKAGE_DIR.parents[1]
DEFAULT_AUDIO_PYTHON = Path(sys.executable)
DEFAULT_MLX_PYTHON = REPOSITORY_DIR / ".venv-mlx" / "bin" / "python"


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=True,
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except Exception as error:  # pragma: no cover - diagnostic only
        return f"unavailable: {error}"


def stage(
    name: str,
    command: list[str],
    *,
    output: Path | None,
    required_outputs: Sequence[Path] = (),
    resume: bool,
    log_dir: Path,
    allowed_codes: set[int] | None = None,
) -> int:
    allowed_codes = allowed_codes or {0}
    outputs = list(required_outputs)
    if output is not None and output not in outputs:
        outputs.insert(0, output)
    marker_dir = log_dir / "completed"
    marker = marker_dir / f"{name}.json"
    # `--resume` only enables a child's internal checkpoint recovery; it does
    # not change the intended artifact.  Excluding it lets a marker created on
    # the first run remain valid on the later pipeline `--resume` invocation.
    artifact_command = [argument for argument in command if argument != "--resume"]
    command_sha256 = hashlib.sha256(
        json.dumps(artifact_command, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    def artifact_is_complete(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    marker_valid = False
    if resume and marker.exists():
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
            marker_valid = (
                state.get("schema_version") == 1
                and state.get("stage") == name
                and state.get("command_sha256") == command_sha256
                and state.get("outputs") == [str(path) for path in outputs]
                and all(artifact_is_complete(path) for path in outputs)
            )
        except (OSError, ValueError, TypeError):
            marker_valid = False
    if marker_valid:
        print(f"[resume] {name}: completed")
        return 0

    # A stale marker must not survive a recovery attempt. Otherwise a process
    # killed after recreating only an early output (for example a partial WAV)
    # could look complete on the following resume.
    marker.unlink(missing_ok=True)
    print(f"[run] {name}", flush=True)
    started = time.time()
    process = subprocess.run(command, text=True, capture_output=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.log").write_text(
        "$ " + " ".join(command) + "\n\n"
        + process.stdout
        + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    elapsed = time.time() - started
    print(f"[done] {name}: {elapsed:.1f}s (exit {process.returncode})", flush=True)
    if process.returncode not in allowed_codes:
        tail = (process.stderr or process.stdout)[-4000:]
        raise RuntimeError(f"Stage {name!r} failed:\n{tail}")
    missing = [path for path in outputs if not artifact_is_complete(path)]
    if missing:
        raise RuntimeError(
            f"Stage {name!r} exited successfully but did not produce: "
            + ", ".join(str(path) for path in missing)
        )

    marker_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "stage": name,
        "command_sha256": command_sha256,
        "outputs": [str(path) for path in outputs],
        "return_code": process.returncode,
        "completed_unix": time.time(),
    }
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)
    return process.returncode


def add_common_script_args(command: list[str], args: argparse.Namespace) -> None:
    if args.aliases:
        command.extend(["--aliases", str(args.aliases)])
    for unit in args.allow_missing_unit:
        command.extend(["--allow-missing-unit", str(unit)])


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_signature(args: argparse.Namespace) -> str:
    payload = {
        "implementation_sha256": implementation_sha256(),
        "audio_sha256": sha256_file(args.audio),
        "script_sha256": sha256_file(args.script),
        "aliases_sha256": sha256_file(args.aliases) if args.aliases else None,
        "allow_missing_unit": sorted(args.allow_missing_unit),
        "language": args.language,
        "mlx_model": args.mlx_model,
        "faster_model": args.faster_model,
        "asr_prompt": args.asr_prompt,
        "ctc": not args.skip_ctc,
        "independent_qa": not args.skip_independent_qa,
        "local_files_only": args.local_files_only,
        "breath_attenuation": not args.disable_breath_attenuation,
        "target_lufs": args.target_lufs,
        "target_peak": args.target_peak,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voicecut",
        description=(
            "Turn a raw scripted narration recording into a verified production WAV."
        ),
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument(
        "--allow-missing-unit",
        type=int,
        action="append",
        default=[],
        help="Explicitly waive a script sentence that was never recorded.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--mlx-model", default="mlx-community/whisper-large-v3-turbo"
    )
    parser.add_argument("--faster-model", default="medium")
    parser.add_argument(
        "--audio-python",
        type=Path,
        default=DEFAULT_AUDIO_PYTHON,
    )
    parser.add_argument(
        "--mlx-python",
        type=Path,
        default=DEFAULT_MLX_PYTHON,
    )
    parser.add_argument("--asr-prompt")
    parser.add_argument("--skip-ctc", action="store_true")
    parser.add_argument("--skip-independent-qa", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download the independent QA model if it is not cached.",
    )
    parser.add_argument("--disable-breath-attenuation", action="store_true")
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--target-peak", type=float, default=-1.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed unless every quality gate passes.",
    )
    parser.add_argument(
        "--accept-review",
        action="store_true",
        help="Publish production.wav on REVIEW (never on FAIL).",
    )
    args = parser.parse_args()

    args.audio = args.audio.resolve()
    args.script = args.script.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.aliases:
        args.aliases = args.aliases.resolve()
    if not args.audio.exists() or not args.script.exists():
        parser.error("Both --audio and --script must exist.")
    if args.aliases and not args.aliases.exists():
        parser.error(f"Alias file not found: {args.aliases}")
    unit_count = len(parse_script(args.script))
    if not unit_count:
        parser.error("The script contains no narratable sentences.")
    invalid_missing = sorted(
        unit
        for unit in args.allow_missing_unit
        if unit < 1 or unit > unit_count
    )
    if invalid_missing:
        parser.error(
            "--allow-missing-unit values are one-based script sentence numbers; "
            f"out of range: {invalid_missing}"
        )
    if not args.audio_python.exists():
        parser.error(f"Audio Python environment not found: {args.audio_python}")
    if not args.mlx_python.exists():
        parser.error(f"MLX Python environment not found: {args.mlx_python}")

    output_dir = args.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output-dir is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        parser.error(
            "--output-dir must be empty for a new run. Choose a new directory "
            "or use --resume with its existing manifest."
        )
    work = output_dir / "work"
    logs = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = run_signature(args)
    manifest_path = output_dir / "run_manifest.json"
    if args.resume and not manifest_path.exists():
        parser.error("--resume requires an existing run_manifest.json.")
    if args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != signature:
            raise RuntimeError(
                "Cannot resume: input, model, or configuration changed. "
                "Choose a new output directory or run without --resume."
            )

    manifest = {
        "schema_version": 1,
        "run_signature": signature,
        "implementation_sha256": implementation_sha256(),
        "audio": str(args.audio),
        "audio_sha256": sha256_file(args.audio),
        "script": str(args.script),
        "script_sha256": sha256_file(args.script),
        "aliases": str(args.aliases) if args.aliases else None,
        "aliases_sha256": sha256_file(args.aliases) if args.aliases else None,
        "language": args.language,
        "mlx_model": args.mlx_model,
        "faster_model": args.faster_model,
        "allow_missing_units": sorted(args.allow_missing_unit),
        "asr_prompt": args.asr_prompt,
        "strict": args.strict,
        "dependencies": {
            "ffmpeg": command_version(["ffmpeg", "-version"]),
            "audio_python": command_version(
                [str(args.audio_python), "--version"]
            ),
            "mlx_python": command_version([str(args.mlx_python), "--version"]),
        },
        "source_probe": probe_audio(args.audio),
        "started_unix": time.time(),
    }
    write_json(manifest_path, manifest)

    source_analysis_dir = work / "source_analysis"
    source_analysis = source_analysis_dir / "analysis.json"
    stage(
        "01_source_analysis",
        [
            str(args.audio_python),
            "-m",
            "voicecut.analyze",
            "--audio",
            str(args.audio),
            "--output-dir",
            str(source_analysis_dir),
        ],
        output=source_analysis,
        required_outputs=[source_analysis_dir / "waveform_features.npz"],
        resume=args.resume,
        log_dir=logs,
    )

    source_transcript = work / "source_transcript.json"
    transcribe_source = [
        str(args.mlx_python),
        "-m",
        "voicecut.transcribe_mlx",
        "--audio",
        str(args.audio),
        "--analysis",
        str(source_analysis),
        "--mode",
        "source",
        "--model",
        args.mlx_model,
        "--language",
        args.language,
        "--output",
        str(source_transcript),
    ]
    if args.asr_prompt:
        transcribe_source.extend(["--prompt", args.asr_prompt])
    if args.resume:
        transcribe_source.append("--resume")
    stage(
        "02_source_asr",
        transcribe_source,
        output=source_transcript,
        resume=args.resume,
        log_dir=logs,
    )

    plan_dir = work / "plan"
    edit_plan = plan_dir / "edit_plan.json"
    plan_command = [
        str(args.audio_python),
        "-m",
        "voicecut.plan",
        "--audio",
        str(args.audio),
        "--script",
        str(args.script),
        "--transcript",
        str(source_transcript),
        "--analysis",
        str(source_analysis),
        "--output-dir",
        str(plan_dir),
    ]
    add_common_script_args(plan_command, args)
    stage(
        "03_edit_plan",
        plan_command,
        output=edit_plan,
        required_outputs=[plan_dir / "ctc_input.json"],
        resume=args.resume,
        log_dir=logs,
    )

    ctc_alignment = plan_dir / "ctc_alignment.json"
    if not args.skip_ctc:
        align_command = [
            str(args.audio_python),
            "-m",
            "voicecut.align_ctc",
            "--audio",
            str(args.audio),
            "--input",
            str(plan_dir / "ctc_input.json"),
            "--output",
            str(ctc_alignment),
            "--language",
            args.language,
        ]
        if args.resume:
            align_command.append("--resume")
        stage(
            "04_ctc_alignment",
            align_command,
            output=ctc_alignment,
            resume=args.resume,
            log_dir=logs,
        )

    edit_dir = work / "edit"
    unmastered = edit_dir / "edited_unmastered.wav"
    render_command = [
        str(args.audio_python),
        "-m",
        "voicecut.render",
        "--audio",
        str(args.audio),
        "--analysis",
        str(source_analysis),
        "--transcript",
        str(source_transcript),
        "--edit-plan",
        str(edit_plan),
        "--output-dir",
        str(edit_dir),
    ]
    if ctc_alignment.exists() and not args.skip_ctc:
        render_command.extend(["--ctc-alignment", str(ctc_alignment)])
    if args.aliases:
        render_command.extend(["--aliases", str(args.aliases)])
    if args.disable_breath_attenuation:
        render_command.append("--disable-breath-attenuation")
    stage(
        "05_render",
        render_command,
        output=unmastered,
        required_outputs=[
            edit_dir / "edit_decision_list.json",
            edit_dir / "edit_decision_list.csv",
            edit_dir / "render_report.json",
        ],
        resume=args.resume,
        log_dir=logs,
    )

    candidate_master = output_dir / "candidate.wav"
    mastering_report = output_dir / "mastering_report.json"
    stage(
        "06_master",
        [
            str(args.audio_python),
            "-m",
            "voicecut.master",
            "--audio",
            str(unmastered),
            "--output",
            str(candidate_master),
            "--report",
            str(mastering_report),
            "--target-lufs",
            str(args.target_lufs),
            "--target-peak",
            str(args.target_peak),
        ],
        output=candidate_master,
        required_outputs=[mastering_report],
        resume=args.resume,
        log_dir=logs,
    )

    qa_analysis_dir = work / "qa_analysis"
    qa_analysis = qa_analysis_dir / "analysis.json"
    stage(
        "07_qa_analysis",
        [
            str(args.audio_python),
            "-m",
            "voicecut.analyze",
            "--audio",
            str(unmastered),
            "--output-dir",
            str(qa_analysis_dir),
        ],
        output=qa_analysis,
        required_outputs=[qa_analysis_dir / "waveform_features.npz"],
        resume=args.resume,
        log_dir=logs,
    )
    qa_primary = work / "qa_primary.json"
    qa_primary_command = [
        str(args.mlx_python),
        "-m",
        "voicecut.transcribe_mlx",
        "--audio",
        str(unmastered),
        "--analysis",
        str(qa_analysis),
        "--mode",
        "source",
        "--model",
        args.mlx_model,
        "--language",
        args.language,
        "--output",
        str(qa_primary),
    ]
    if args.asr_prompt:
        qa_primary_command.extend(["--prompt", args.asr_prompt])
    if args.resume:
        qa_primary_command.append("--resume")
    stage(
        "08_qa_primary_asr",
        qa_primary_command,
        output=qa_primary,
        resume=args.resume,
        log_dir=logs,
    )

    qa_independent = work / "qa_independent.json"
    if not args.skip_independent_qa:
        independent_command = [
            str(args.audio_python),
            "-m",
            "voicecut.transcribe_faster",
            "--audio",
            str(unmastered),
            "--output",
            str(qa_independent),
            "--model",
            args.faster_model,
            "--language",
            args.language,
        ]
        if args.local_files_only:
            independent_command.append("--local-files-only")
        stage(
            "09_qa_independent_asr",
            independent_command,
            output=qa_independent,
            resume=args.resume,
            log_dir=logs,
        )

    qa_report = output_dir / "qa_report.json"
    qa_command = [
        str(args.audio_python),
        "-m",
        "voicecut.qa",
        "--audio",
        str(unmastered),
        "--script",
        str(args.script),
        "--transcript",
        str(qa_primary),
        "--edit-plan",
        str(edit_plan),
        "--edl",
        str(edit_dir / "edit_decision_list.json"),
        "--mastered",
        str(candidate_master),
        "--output",
        str(qa_report),
        "--target-lufs",
        str(args.target_lufs),
        "--target-true-peak",
        str(args.target_peak),
    ]
    add_common_script_args(qa_command, args)
    if qa_independent.exists() and not args.skip_independent_qa:
        qa_command.extend(["--independent-transcript", str(qa_independent)])
    if args.strict:
        qa_command.append("--strict")
    stage(
        "10_quality_gate",
        qa_command,
        output=qa_report,
        resume=False,
        log_dir=logs,
        allowed_codes={0, 2},
    )
    qa = json.loads(qa_report.read_text(encoding="utf-8"))
    verdict = str(qa["verdict"])

    # Surface the auditable artifacts at the run root.
    shutil.copy2(edit_plan, output_dir / "edit_plan.json")
    shutil.copy2(
        edit_dir / "edit_decision_list.json",
        output_dir / "edit_decision_list.json",
    )
    review_source = qa_report.parent / "review_clips"
    if review_source.exists() and review_source != output_dir / "review_clips":
        shutil.copytree(
            review_source,
            output_dir / "review_clips",
            dirs_exist_ok=True,
        )

    publish = verdict == "PASS" or (
        verdict == "REVIEW" and args.accept_review
    )
    production = output_dir / "production.wav"
    if publish:
        if production.exists():
            production.unlink()
        candidate_master.rename(production)
        final_audio = production
    else:
        if production.exists():
            previous = output_dir / "production.previous.wav"
            suffix = 1
            while previous.exists():
                previous = output_dir / f"production.previous.{suffix}.wav"
                suffix += 1
            production.rename(previous)
            manifest["quarantined_previous_production"] = str(previous)
        final_audio = candidate_master

    manifest["finished_unix"] = time.time()
    manifest["verdict"] = verdict
    manifest["published"] = publish
    manifest["final_audio"] = str(final_audio)
    manifest["qa_report"] = str(qa_report)
    write_json(manifest_path, manifest)
    result = {
        "verdict": verdict,
        "published": publish,
        "audio": str(final_audio),
        "qa_report": str(qa_report),
        "edit_plan": str(output_dir / "edit_plan.json"),
        "edl": str(output_dir / "edit_decision_list.json"),
        "manifest": str(manifest_path),
    }
    write_json(output_dir / "pipeline_result.json", result)
    print(json.dumps(result, indent=2))
    if args.strict and not publish:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
