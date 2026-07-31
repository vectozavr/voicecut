#!/usr/bin/env python3
"""The single production entry point for VoiceCut.

The public pipeline accepts an audio or video file, creates one canonical WAV,
builds a source-grounded semantic edit plan, renders safe word boundaries and
semantic pauses, and publishes the requested media format. Expensive completed
stages are content-addressed and reused on an identical rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file, write_json
from .media import (
    AUDIO_OUTPUT_EXTENSIONS,
    MediaError,
    VIDEO_INPUT_EXTENSIONS,
    VIDEO_OUTPUT_EXTENSIONS,
    output_media_kind,
    prepare_media_input,
    publish_audio,
)
from .mfa_alignment import (
    DEFAULT_MFA_CACHE_ROOT,
    DEFAULT_MFA_PREFIX,
    MFA_MODEL_ID,
    MFA_VERSION,
)
from .planner_backends import (
    API_KEY_ENV_BY_BACKEND,
    DEFAULT_MAX_OUTPUT_TOKENS,
    PlannerRuntimeConfiguration,
    add_planner_backend_arguments,
    default_model_for_backend,
    preflight_planner_backend,
    resolve_planner_base_url,
)
from .video_render import render_edited_video


PACKAGE_PARENT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_ALIGNMENT_BACKEND = "mfa"
MFA_MODEL = MFA_MODEL_ID
MFA_FINE_TUNE = True
DEFAULT_MFA_MICROMAMBA = Path("micromamba")
DEFAULT_MFA_NUM_JOBS = max(1, min(os.cpu_count() or 1, 4))
PIPELINE_SCHEMA_VERSION = 5


def _preferred_python(relative_path: str) -> Path:
    candidate = PROJECT_ROOT / relative_path
    return candidate if candidate.is_file() else Path(sys.executable)


DEFAULT_ASR_PYTHON = _preferred_python(".venv-mlx/bin/python")
DEFAULT_ALIGNMENT_PYTHON = Path(sys.executable)


class FullPipelineError(RuntimeError):
    """The one-command production pipeline cannot safely continue."""


def _implementation_fingerprint() -> str:
    """Hash production source so a software change cannot reuse stale stages."""

    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _WorkDirectoryLock:
    """Hold a non-blocking process lock for one cache directory."""

    def __init__(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        self.path = work_dir / ".voicecut.lock"
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "another process"
            self._handle.close()
            raise FullPipelineError(
                f"another VoiceCut run owns this cache ({owner}): {work_dir}"
            ) from None
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or handle.closed:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def _audio_artifact_is_current(path: Path, audio_sha256: str) -> bool:
    value = _read_object(path)
    return value is not None and value.get("audio_sha256") == audio_sha256


def _media_is_current(
    manifest_path: Path,
    *,
    source_path: Path,
    source_sha256: str,
) -> bool:
    value = _read_object(manifest_path)
    if value is None:
        return False
    audio_value = value.get("canonical_audio")
    if not isinstance(audio_value, str):
        return False
    audio_path = Path(audio_value)
    expected_audio_sha = value.get("canonical_audio_sha256")
    return (
        value.get("status") == "complete"
        and value.get("source_media") == str(source_path)
        and value.get("source_media_sha256") == source_sha256
        and audio_path.is_file()
        and isinstance(expected_audio_sha, str)
        and sha256_file(audio_path) == expected_audio_sha
    )


def _enriched_transcript_is_current(path: Path, audio_sha256: str) -> bool:
    value = _read_object(path)
    return (
        value is not None
        and value.get("audio_sha256") == audio_sha256
        and value.get("source_decode_strategy")
        == "whisper_primary_plus_gated_raw_ctc_insertions_v1"
    )


def _plan_is_current(
    path: Path,
    *,
    transcript_path: Path,
    backend: str,
    model: str,
) -> bool:
    value = _read_object(path)
    return (
        value is not None
        and value.get("status") == "complete"
        and value.get("backend") == backend
        and value.get("model") == model
        and value.get("transcript") == str(transcript_path.resolve())
        and value.get("transcript_sha256") == sha256_file(transcript_path)
    )


def _render_is_current(
    manifest_path: Path,
    *,
    audio_sha256: str,
    plan_path: Path,
    backend: str,
    model: str,
) -> bool:
    value = _read_object(manifest_path)
    if value is None:
        return False
    final_cut = value.get("final_cut_wav")
    final_cut_sha = value.get("final_cut_wav_sha256")
    boundary_plan = value.get("final_boundary_plan")
    boundary_plan_sha = value.get("final_boundary_plan_sha256")
    effective_plan = value.get("effective_streaming_plan")
    effective_plan_sha = value.get("effective_streaming_plan_sha256")
    return (
        value.get("status") == "complete"
        and value.get("renderer") == "authoritative_single_pass_final_render_v3"
        and value.get("alignment_backend") == DEFAULT_ALIGNMENT_BACKEND
        and value.get("mfa_version") == MFA_VERSION
        and value.get("mfa_model") == MFA_MODEL
        and value.get("mfa_fine_tune") is MFA_FINE_TUNE
        and value.get("source_audio_sha256") == audio_sha256
        and value.get("streaming_plan") == str(plan_path.resolve())
        and value.get("streaming_plan_sha256") == sha256_file(plan_path)
        and value.get("pause_planner_backend") == backend
        and value.get("pause_planner_model") == model
        and isinstance(final_cut, str)
        and Path(final_cut).is_file()
        and isinstance(final_cut_sha, str)
        and sha256_file(Path(final_cut)) == final_cut_sha
        and isinstance(boundary_plan, str)
        and Path(boundary_plan).is_file()
        and isinstance(boundary_plan_sha, str)
        and sha256_file(Path(boundary_plan)) == boundary_plan_sha
        and isinstance(effective_plan, str)
        and Path(effective_plan).is_file()
        and isinstance(effective_plan_sha, str)
        and sha256_file(Path(effective_plan)) == effective_plan_sha
    )


def _publication_is_current(
    manifest_path: Path,
    *,
    output_path: Path,
    final_manifest_path: Path,
) -> bool:
    value = _read_object(manifest_path)
    if value is None or not output_path.is_file():
        return False
    return (
        value.get("status") == "complete"
        and value.get("output") == str(output_path)
        and value.get("output_sha256") == sha256_file(output_path)
        and value.get("final_render_manifest_sha256")
        == sha256_file(final_manifest_path)
    )


def _reset_stage(directory: Path, *, work_dir: Path) -> None:
    """Remove only an incomplete pipeline-owned stage directory."""

    if not directory.exists():
        return
    resolved = directory.resolve()
    if resolved.parent != work_dir.resolve():
        raise FullPipelineError(f"refusing to reset a directory outside {work_dir}")
    shutil.rmtree(resolved)


def _subprocess_environment(
    *,
    planner_backend: str | None = None,
    planner_api_key_env: str | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    credential_names = set(API_KEY_ENV_BY_BACKEND.values())
    if planner_api_key_env:
        credential_names.add(planner_api_key_env)
    allowed_credential = None
    if planner_backend in API_KEY_ENV_BY_BACKEND:
        allowed_credential = (
            planner_api_key_env or API_KEY_ENV_BY_BACKEND[planner_backend]
        )
    for name in credential_names:
        if name != allowed_credential:
            environment.pop(name, None)
    # Effective provider endpoints are validated and passed explicitly. Do not
    # leak unrelated endpoint configuration into local DSP/ASR subprocesses.
    environment.pop("OPENAI_BASE_URL", None)
    environment.pop("DEEPSEEK_BASE_URL", None)
    package_parent = str(PACKAGE_PARENT)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        package_parent if not existing else package_parent + os.pathsep + existing
    )
    return environment


def _redact_planner_endpoint(value: str, command: Sequence[str]) -> str:
    redacted = value
    for index, argument in enumerate(command):
        endpoint: str | None = None
        if argument == "--planner-base-url" and index + 1 < len(command):
            endpoint = command[index + 1]
        elif argument.startswith("--planner-base-url="):
            endpoint = argument.partition("=")[2]
        if endpoint:
            redacted = redacted.replace(endpoint, "<redacted-endpoint>")
    return redacted


def _concise_subprocess_failure(
    stderr: str | bytes | None,
    *,
    command: Sequence[str],
) -> str | None:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if not isinstance(stderr, str):
        return None
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if (
            not stripped
            or stripped == "Traceback (most recent call last):"
            or stripped.startswith('File "')
            or set(stripped) <= {"^", "~"}
        ):
            continue
        safe = _redact_planner_endpoint(stripped, command)
        return safe if len(safe) <= 800 else "…" + safe[-799:]
    return None


def _run(
    stage: str,
    command: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    planner_backend: str | None = None,
    planner_api_key_env: str | None = None,
) -> None:
    print(f"\n=== {stage} ===", flush=True)
    visible_command = list(command)
    for index, value in enumerate(visible_command[:-1]):
        if value == "--planner-base-url":
            visible_command[index + 1] = "<redacted-endpoint>"
    visible_command = [
        (
            "--planner-base-url=<redacted-endpoint>"
            if value.startswith("--planner-base-url=")
            else value
        )
        for value in visible_command
    ]
    print("$ " + shlex.join(visible_command), flush=True)
    try:
        completed = runner(
            command,
            check=True,
            cwd=PROJECT_ROOT,
            stderr=subprocess.PIPE,
            text=True,
            env=_subprocess_environment(
                planner_backend=planner_backend,
                planner_api_key_env=planner_api_key_env,
            ),
        )
    except subprocess.CalledProcessError as error:
        detail = _concise_subprocess_failure(error.stderr, command=command)
        suffix = f": {detail}" if detail else ""
        raise FullPipelineError(
            f"{stage} failed with exit status {error.returncode}{suffix}"
        ) from None
    except OSError as error:
        detail = _redact_planner_endpoint(str(error), command)
        raise FullPipelineError(f"{stage} could not start: {detail}") from None
    child_stderr = getattr(completed, "stderr", None)
    if isinstance(child_stderr, str) and child_stderr:
        print(
            _redact_planner_endpoint(child_stderr, command),
            end="" if child_stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )


def _configuration_digest(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return compact[:48] or "run"


def _default_output(input_path: Path) -> Path:
    extension = input_path.suffix.lower()
    if extension not in AUDIO_OUTPUT_EXTENSIONS | VIDEO_OUTPUT_EXTENSIONS:
        extension = ".mp4" if extension in VIDEO_INPUT_EXTENSIONS else ".wav"
    return input_path.with_name(f"{input_path.stem}_edited{extension}")


def _resolve_executable(value: Path, *, label: str) -> Path:
    """Resolve a configured executable without relying on a child shell."""

    expanded = value.expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        resolved = expanded.absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FullPipelineError(f"{label} is not executable: {resolved}")
        return resolved
    discovered = shutil.which(str(expanded))
    if discovered is None:
        raise FullPipelineError(
            f"{label} was not found on PATH: {expanded}; run scripts/install.sh"
        )
    # Preserve the configured command name/symlink (notably the Homebrew
    # `micromamba` link) so provenance records the CLI that was invoked.
    return Path(discovered).absolute()


def _validate_mfa_runtime(
    *,
    backend: str,
    prefix: Path,
    cache_root: Path,
    micromamba: Path,
    num_jobs: int,
) -> tuple[Path, Path, Path]:
    """Validate the pinned MFA runtime paths before any media work starts."""

    if backend != DEFAULT_ALIGNMENT_BACKEND:
        raise FullPipelineError(
            "MFA is the sole production alignment backend; "
            f"unsupported --alignment-backend: {backend}"
        )
    if num_jobs <= 0:
        raise FullPipelineError("--mfa-num-jobs must be positive")
    resolved_prefix = prefix.expanduser().absolute()
    if not resolved_prefix.is_dir():
        raise FullPipelineError(
            f"MFA environment prefix does not exist: {resolved_prefix}; "
            "run scripts/install.sh"
        )
    mfa_executable = resolved_prefix / "bin" / "mfa"
    if not mfa_executable.is_file() or not os.access(mfa_executable, os.X_OK):
        raise FullPipelineError(
            "MFA executable is missing from the configured prefix: "
            f"{mfa_executable}; run scripts/install.sh"
        )
    resolved_micromamba = _resolve_executable(
        micromamba,
        label="micromamba executable",
    )
    resolved_cache_root = cache_root.expanduser().absolute()
    if resolved_cache_root.exists() and not resolved_cache_root.is_dir():
        raise FullPipelineError(
            f"MFA cache root is not a directory: {resolved_cache_root}"
        )
    resolved_cache_root.mkdir(parents=True, exist_ok=True)
    return resolved_prefix, resolved_cache_root, resolved_micromamba


def _configuration(
    *,
    input_path: Path,
    input_sha256: str,
    args: argparse.Namespace,
    planner_model: str,
    planner_base_url: str | None,
    asr_python: Path,
    alignment_python: Path,
    planner_python: Path,
    mfa_prefix: Path,
    mfa_cache_root: Path,
    mfa_micromamba: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "implementation_sha256": _implementation_fingerprint(),
        "input": str(input_path),
        "input_sha256": input_sha256,
        "language": args.language,
        "whisper_model": args.whisper_model,
        "planner_backend": args.planner_backend,
        "planner_model": planner_model,
        "planner_base_url": planner_base_url,
        "planner_api_key_env": args.planner_api_key_env,
        "planner_local_files_only": bool(args.local_files_only),
        "window_seconds": float(args.window_seconds),
        "max_output_tokens": int(args.max_output_tokens),
        "max_acoustic_retries": int(args.max_acoustic_retries),
        "debug_artifacts": bool(args.debug_artifacts),
        "asr_python": str(asr_python),
        "alignment_backend": args.alignment_backend,
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL,
        "mfa_fine_tune": MFA_FINE_TUNE,
        "mfa_prefix": str(mfa_prefix),
        "mfa_cache_root": str(mfa_cache_root),
        "mfa_micromamba": str(mfa_micromamba),
        "mfa_num_jobs": int(args.mfa_num_jobs),
        # WhisperX remains only as a retained-word completeness veto. It is
        # not permitted to provide production source-sample coordinates.
        "alignment_python": str(alignment_python),
        "alignment_python_role": "whisperx_retained_word_completeness_veto_only",
        "planner_python": str(planner_python),
    }


def _resolve_work_dir(
    *,
    requested: Path | None,
    output_path: Path,
    input_path: Path,
    configuration: dict[str, Any],
) -> Path:
    if requested is not None:
        return requested.resolve()
    digest = _configuration_digest(configuration)[:12]
    label = _safe_name(f"{input_path.stem}-{configuration['planner_backend']}")
    return (output_path.parent / ".voicecut-cache" / f"{label}-{digest}").resolve()


def _paths_overlap_work_tree(file_path: Path, work_dir: Path) -> bool:
    """Return true when a user media path and the cache contain one another."""

    return (
        file_path == work_dir
        or work_dir in file_path.parents
        or file_path in work_dir.parents
    )


def _initialize_work_dir(
    work_dir: Path,
    *,
    configuration: dict[str, Any],
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "pipeline_config.json"
    existing = _read_object(path)
    if existing is not None and existing != configuration:
        raise FullPipelineError(
            f"{work_dir} belongs to a different input or configuration; "
            "choose another --work-dir"
        )
    if existing is None:
        if path.exists():
            raise FullPipelineError(f"invalid pipeline configuration file: {path}")
        write_json(path, configuration)
    return path


def _planner_cli_arguments(
    *,
    args: argparse.Namespace,
    planner_model: str,
    planner_base_url: str | None,
    planner_python: Path,
    env_file: Path,
) -> list[str]:
    values = [
        "--planner-backend",
        args.planner_backend,
        "--planner-model",
        planner_model,
        "--planner-python",
        str(planner_python),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--env-file",
        str(env_file),
    ]
    if planner_base_url:
        values.extend(["--planner-base-url", planner_base_url])
    if args.planner_api_key_env:
        values.extend(["--planner-api-key-env", args.planner_api_key_env])
    if args.local_files_only:
        values.append("--local-files-only")
    return values


def _publication_record(
    *,
    kind: str,
    output_path: Path,
    final_manifest_path: Path,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "kind": kind,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "final_render_manifest": str(final_manifest_path),
        "final_render_manifest_sha256": sha256_file(final_manifest_path),
        "details": details,
    }


def run_full_pipeline(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    media_preparer: Callable[..., dict[str, Any]] = prepare_media_input,
    audio_publisher: Callable[..., dict[str, Any]] = publish_audio,
    video_publisher: Callable[..., dict[str, Any]] = render_edited_video,
    planner_preflight: Callable[..., PlannerRuntimeConfiguration] = (
        preflight_planner_backend
    ),
) -> dict[str, Any]:
    """Run or resume the complete pipeline and return its provenance ledger."""

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = (
        args.output.absolute()
        if args.output is not None
        else _default_output(input_path).absolute()
    )
    if output_path.is_symlink():
        raise FullPipelineError(
            f"output path must not be a symbolic link: {output_path}"
        )
    output_path = output_path.resolve()
    if output_path == input_path:
        raise FullPipelineError("input and output paths must be different")

    try:
        output_kind = output_media_kind(output_path)
    except MediaError as error:
        raise FullPipelineError(str(error)) from error
    if args.window_seconds <= 0:
        raise FullPipelineError("--window-seconds must be positive")
    if args.max_output_tokens <= 0:
        raise FullPipelineError("--max-output-tokens must be positive")

    asr_python = args.asr_python.absolute()
    alignment_python = args.alignment_python.absolute()
    planner_python = args.planner_python.absolute()
    if not asr_python.is_file():
        raise FullPipelineError(f"ASR Python does not exist: {asr_python}")
    if not alignment_python.is_file():
        raise FullPipelineError(
            f"WhisperX completeness-veto Python does not exist: {alignment_python}"
        )
    mfa_prefix, mfa_cache_root, mfa_micromamba = _validate_mfa_runtime(
        backend=args.alignment_backend,
        prefix=args.mfa_prefix,
        cache_root=args.mfa_cache_root,
        micromamba=args.mfa_micromamba,
        num_jobs=args.mfa_num_jobs,
    )
    if args.planner_backend in {"local", "qwen", "gemma"}:
        if not planner_python.is_file() and asr_python.is_file():
            planner_python = asr_python
        if not planner_python.is_file():
            raise FullPipelineError(
                f"local planner Python does not exist: {planner_python}"
            )

    env_file = args.env_file.absolute()
    planner_model = args.planner_model or default_model_for_backend(
        args.planner_backend
    )
    try:
        planner_base_url = resolve_planner_base_url(
            provider=args.planner_backend,
            env_file=env_file,
            base_url=args.planner_base_url,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise FullPipelineError(f"planner configuration is invalid: {error}") from None
    input_sha256 = sha256_file(input_path)
    configuration = _configuration(
        input_path=input_path,
        input_sha256=input_sha256,
        args=args,
        planner_model=planner_model,
        planner_base_url=planner_base_url,
        asr_python=asr_python,
        alignment_python=alignment_python,
        planner_python=planner_python,
        mfa_prefix=mfa_prefix,
        mfa_cache_root=mfa_cache_root,
        mfa_micromamba=mfa_micromamba,
    )
    work_dir = _resolve_work_dir(
        requested=args.work_dir,
        output_path=output_path,
        input_path=input_path,
        configuration=configuration,
    )
    for label, media_path in (("input", input_path), ("output", output_path)):
        if _paths_overlap_work_tree(media_path, work_dir):
            raise FullPipelineError(
                f"{label} path and --work-dir must be disjoint: "
                f"{media_path} versus {work_dir}"
            )
    try:
        planner_runtime = planner_preflight(
            provider=args.planner_backend,
            env_file=env_file,
            base_url=planner_base_url,
            api_key_env=args.planner_api_key_env,
            local_python=planner_python,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise FullPipelineError(f"planner preflight failed: {error}") from None
    if planner_runtime.base_url != planner_base_url:
        raise FullPipelineError(
            "planner preflight returned an inconsistent API endpoint"
        )
    work_lock = _WorkDirectoryLock(work_dir)
    configuration_path = _initialize_work_dir(
        work_dir,
        configuration=configuration,
    )
    stages: dict[str, str] = {}

    media_dir = work_dir / "00_media"
    media_manifest_path = media_dir / "media_input.json"
    if _media_is_current(
        media_manifest_path,
        source_path=input_path,
        source_sha256=input_sha256,
    ):
        media_manifest = _read_object(media_manifest_path)
        assert media_manifest is not None
        stages["media_input"] = "cached"
    else:
        _reset_stage(media_dir, work_dir=work_dir)
        print("\n=== 00 MEDIA INPUT ===", flush=True)
        media_manifest = media_preparer(input_path, media_dir)
        if not _media_is_current(
            media_manifest_path,
            source_path=input_path,
            source_sha256=input_sha256,
        ):
            raise FullPipelineError(
                "media preparation did not produce a current canonical WAV"
            )
        stages["media_input"] = "created"

    source_kind = media_manifest.get("source_kind")
    if source_kind not in {"audio", "video"}:
        raise FullPipelineError("media input has no supported source kind")
    if output_kind == "video" and source_kind != "video":
        raise FullPipelineError(
            "a video output requires a video input; choose an audio extension"
        )
    audio_value = media_manifest.get("canonical_audio")
    audio_sha256 = media_manifest.get("canonical_audio_sha256")
    if not isinstance(audio_value, str) or not isinstance(audio_sha256, str):
        raise FullPipelineError("media input manifest has no canonical audio")
    audio = Path(audio_value)

    analysis_dir = work_dir / "01_analysis"
    analysis_path = analysis_dir / "analysis.json"
    if _audio_artifact_is_current(analysis_path, audio_sha256):
        stages["analysis"] = "cached"
    else:
        _reset_stage(analysis_dir, work_dir=work_dir)
        _run(
            "01 ANALYSIS",
            [
                sys.executable,
                "-m",
                "voicecut.analyze",
                "--audio",
                str(audio),
                "--output-dir",
                str(analysis_dir),
            ],
            runner=runner,
        )
        if not _audio_artifact_is_current(analysis_path, audio_sha256):
            raise FullPipelineError("analysis did not produce a current analysis.json")
        stages["analysis"] = "created"

    transcript_dir = work_dir / "02_transcription"
    transcript_path = transcript_dir / "source_transcript.json"
    if _audio_artifact_is_current(transcript_path, audio_sha256):
        stages["transcription"] = "cached"
    else:
        _reset_stage(transcript_dir, work_dir=work_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        _run(
            "02 WHISPER TRANSCRIPTION",
            [
                str(asr_python),
                "-m",
                "voicecut.transcribe_mlx",
                "--audio",
                str(audio),
                "--analysis",
                str(analysis_path),
                "--mode",
                "source",
                "--skip-whole",
                "--model",
                args.whisper_model,
                "--language",
                args.language,
                "--artifact-role",
                "source_primary",
                "--output",
                str(transcript_path),
                "--resume",
            ],
            runner=runner,
        )
        if not _audio_artifact_is_current(transcript_path, audio_sha256):
            raise FullPipelineError(
                "Whisper did not produce a current source transcript"
            )
        stages["transcription"] = "created"

    enrichment_dir = work_dir / "03_ctc_enrichment"
    enriched_path = enrichment_dir / "source_transcript_ctc_enriched.json"
    if _enriched_transcript_is_current(enriched_path, audio_sha256):
        stages["hidden_retry_recovery"] = "cached"
    else:
        _reset_stage(enrichment_dir, work_dir=work_dir)
        _run(
            "03 ACOUSTIC HIDDEN-RETRY RECOVERY",
            [
                sys.executable,
                "-m",
                "voicecut.ctc_enrich",
                "--audio",
                str(audio),
                "--transcript",
                str(transcript_path),
                "--output-dir",
                str(enrichment_dir),
            ],
            runner=runner,
        )
        if not _enriched_transcript_is_current(enriched_path, audio_sha256):
            raise FullPipelineError(
                "CTC recovery did not produce a current enriched transcript"
            )
        stages["hidden_retry_recovery"] = "created"

    planner_arguments = _planner_cli_arguments(
        args=args,
        planner_model=planner_model,
        planner_base_url=planner_runtime.base_url,
        planner_python=planner_python,
        env_file=env_file,
    )
    plan_dir = work_dir / "04_semantic_plan"
    plan_path = plan_dir / "streaming_plan.json"
    if _plan_is_current(
        plan_path,
        transcript_path=enriched_path,
        backend=args.planner_backend,
        model=planner_model,
    ):
        stages["semantic_plan"] = "cached"
    else:
        _reset_stage(plan_dir, work_dir=work_dir)
        _run(
            "04 STREAMING SEMANTIC PLAN",
            [
                sys.executable,
                "-m",
                "voicecut.streaming_narration",
                "--stream-plan",
                "--transcript",
                str(enriched_path),
                "--output-dir",
                str(plan_dir),
                "--window-seconds",
                str(args.window_seconds),
                *planner_arguments,
            ],
            runner=runner,
            planner_backend=planner_runtime.provider,
            planner_api_key_env=planner_runtime.api_key_env,
        )
        if not _plan_is_current(
            plan_path,
            transcript_path=enriched_path,
            backend=args.planner_backend,
            model=planner_model,
        ):
            raise FullPipelineError(
                "the semantic planner did not produce a valid grounded plan"
            )
        stages["semantic_plan"] = "created"

    final_dir = work_dir / "05_final"
    final_manifest_path = final_dir / "final_render_manifest.json"
    if _render_is_current(
        final_manifest_path,
        audio_sha256=audio_sha256,
        plan_path=plan_path,
        backend=args.planner_backend,
        model=planner_model,
    ):
        stages["final_render"] = "cached"
    else:
        _reset_stage(final_dir, work_dir=work_dir)
        render_command = [
            sys.executable,
            "-m",
            "voicecut.final_render",
            "--render-plan",
            "--audio",
            str(audio),
            "--plan",
            str(plan_path),
            "--output-dir",
            str(final_dir),
            "--alignment-python",
            str(alignment_python),
            "--alignment-backend",
            args.alignment_backend,
            "--mfa-prefix",
            str(mfa_prefix),
            "--mfa-cache-root",
            str(mfa_cache_root),
            "--mfa-micromamba",
            str(mfa_micromamba),
            "--mfa-num-jobs",
            str(args.mfa_num_jobs),
            "--max-acoustic-retries",
            str(args.max_acoustic_retries),
            *planner_arguments,
        ]
        if args.debug_artifacts:
            render_command.append("--debug-artifacts")
        _run(
            "05 AUTHORITATIVE SINGLE-PASS RENDER",
            render_command,
            runner=runner,
            planner_backend=planner_runtime.provider,
            planner_api_key_env=planner_runtime.api_key_env,
        )
        if not _render_is_current(
            final_manifest_path,
            audio_sha256=audio_sha256,
            plan_path=plan_path,
            backend=args.planner_backend,
            model=planner_model,
        ):
            raise FullPipelineError(
                "the final renderer did not produce a validated final cut"
            )
        stages["final_render"] = "created"

    final_manifest = _read_object(final_manifest_path)
    if final_manifest is None:
        raise FullPipelineError("final render manifest is missing")
    final_wav_value = final_manifest.get("final_cut_wav")
    if not isinstance(final_wav_value, str):
        raise FullPipelineError("final render manifest has no WAV output")
    final_wav = Path(final_wav_value)

    publication_dir = work_dir / "06_publication"
    publication_path = publication_dir / "publication.json"
    if _publication_is_current(
        publication_path,
        output_path=output_path,
        final_manifest_path=final_manifest_path,
    ):
        publication = _read_object(publication_path)
        assert publication is not None
        stages["publication"] = "cached"
    else:
        _reset_stage(publication_dir, work_dir=work_dir)
        publication_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists; pass --overwrite to replace it"
            )
        print("\n=== 06 MEDIA PUBLICATION ===", flush=True)
        if output_kind == "video":
            video_manifest_path = publication_dir / "video_render_manifest.json"
            details = video_publisher(
                source_video=input_path,
                media_input_manifest_path=media_manifest_path,
                final_render_manifest_path=final_manifest_path,
                output_path=output_path,
                manifest_path=video_manifest_path,
                overwrite=args.overwrite,
            )
            kind = "video"
        else:
            details = audio_publisher(
                final_wav,
                output_path,
                manifest_path=publication_dir / "audio_publish_manifest.json",
                overwrite=args.overwrite,
            )
            kind = "audio"
        publication = _publication_record(
            kind=kind,
            output_path=output_path,
            final_manifest_path=final_manifest_path,
            details=details,
        )
        write_json(publication_path, publication)
        stages["publication"] = "created"

    result = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "complete",
        "input": str(input_path),
        "input_sha256": input_sha256,
        "input_kind": source_kind,
        "output": str(output_path),
        "output_sha256": publication["output_sha256"],
        "output_kind": publication["kind"],
        "work_dir": str(work_dir),
        "pipeline_configuration": str(configuration_path),
        "planner_backend": args.planner_backend,
        "planner_model": planner_model,
        "alignment_backend": args.alignment_backend,
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL,
        "stages": stages,
        "media_input_manifest": str(media_manifest_path),
        "analysis": str(analysis_path),
        "transcript": str(transcript_path),
        "enriched_transcript": str(enriched_path),
        "streaming_plan": str(plan_path),
        "render_manifest": str(final_manifest_path),
        "publication_manifest": str(publication_path),
        "final_cut_wav": str(final_wav),
        "duration_seconds": final_manifest["duration_seconds"],
    }
    write_json(work_dir / "pipeline_run.json", result)
    work_lock.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicecut",
        description=(
            "Turn a retake-heavy narration recording into a coherent edited "
            "audio or video file with one command."
        ),
    )
    parser.add_argument("input", type=Path, help="Input audio or video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Published output path. The extension selects the format; defaults "
            "to INPUT_STEM_edited with the input's common media extension."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "Persistent stage cache. Defaults to a content-addressed directory "
            "under .voicecut-cache beside the output."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output that is not a valid cached result.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--asr-python",
        type=Path,
        default=DEFAULT_ASR_PYTHON,
        help="Python executable containing mlx-whisper.",
    )
    parser.add_argument(
        "--alignment-python",
        type=Path,
        default=DEFAULT_ALIGNMENT_PYTHON,
        help=(
            "Python executable containing WhisperX for the retained-word "
            "completeness veto only; it never supplies production cut "
            "coordinates."
        ),
    )
    parser.add_argument(
        "--alignment-backend",
        choices=(DEFAULT_ALIGNMENT_BACKEND,),
        default=DEFAULT_ALIGNMENT_BACKEND,
        help=(
            "Production cut-coordinate backend. MFA is authoritative; "
            "Whisper timestamps remain approximate crop anchors only."
        ),
    )
    parser.add_argument(
        "--mfa-prefix",
        type=Path,
        default=DEFAULT_MFA_PREFIX,
        help="Repository-local micromamba prefix containing MFA 3.4.1.",
    )
    parser.add_argument(
        "--mfa-cache-root",
        type=Path,
        default=DEFAULT_MFA_CACHE_ROOT,
        help="Persistent MFA model/cache directory passed as MFA_ROOT_DIR.",
    )
    parser.add_argument(
        "--mfa-micromamba",
        type=Path,
        default=DEFAULT_MFA_MICROMAMBA,
        help="micromamba executable used to invoke the pinned MFA prefix.",
    )
    parser.add_argument(
        "--mfa-num-jobs",
        type=int,
        default=DEFAULT_MFA_NUM_JOBS,
        help="Parallel MFA jobs used by the single batched align_hf invocation.",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument(
        "--language",
        choices=("en",),
        default="en",
        help="Source language. This production release currently supports English.",
    )
    add_planner_backend_arguments(parser)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Streaming semantic look-ahead increment; never an edit boundary.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--max-acoustic-retries",
        type=int,
        default=3,
        help=(
            "Maximum planner reselections after alignment finds an "
            "uncuttable dense boundary or a weak retained-word occurrence."
        ),
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help=(
            "Request optional diagnostics without changing the single-pass "
            "production render graph."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_acoustic_retries < 0:
        parser.error("--max-acoustic-retries must be non-negative")
    try:
        result = run_full_pipeline(args)
    except (FullPipelineError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print("\nVOICECUT COMPLETE")
    print(f"output: {result['output']}")
    print(f"duration: {result['duration_seconds']:.3f} s")
    print(f"planner: {result['planner_backend']} / {result['planner_model']}")
    print(f"cache: {result['work_dir']}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
