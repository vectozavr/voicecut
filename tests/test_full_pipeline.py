from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voicecut.common import read_json, sha256_file, write_json
from voicecut.full_pipeline import (
    FullPipelineError,
    _WorkDirectoryLock,
    _implementation_fingerprint,
    _run,
    _subprocess_environment,
    build_parser,
    run_full_pipeline,
)
from voicecut.planner_backends import PlannerRuntimeConfiguration


def _args(tmp_path: Path, source: Path):
    return build_parser().parse_args(
        [
            str(source),
            "--output",
            str(tmp_path / "edited.mp3"),
            "--work-dir",
            str(tmp_path / "work"),
            "--asr-python",
            sys.executable,
            "--alignment-python",
            sys.executable,
            "--planner-python",
            sys.executable,
            "--planner-backend",
            "gemini",
            "--planner-model",
            "gemini-3.6-flash",
        ]
    )


def _fake_gemini_preflight(**_: object) -> PlannerRuntimeConfiguration:
    return PlannerRuntimeConfiguration(
        provider="gemini",
        base_url=None,
        api_key_env="GEMINI_API_KEY",
    )


def test_full_pipeline_runs_every_stage_and_then_uses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source media")
    args = _args(tmp_path, source)
    calls: list[list[str]] = []
    publication_calls = 0
    monkeypatch.setenv("GEMINI_API_KEY", "selected-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unrelated-secret")

    def fake_media_preparer(input_path: Path, output_dir: Path):
        output_dir.mkdir(parents=True)
        canonical = output_dir / "source_audio.wav"
        canonical.write_bytes(b"canonical audio")
        manifest = {
            "status": "complete",
            "source_media": str(input_path.resolve()),
            "source_media_sha256": sha256_file(input_path),
            "source_kind": "audio",
            "canonical_audio": str(canonical.resolve()),
            "canonical_audio_sha256": sha256_file(canonical),
        }
        write_json(output_dir / "media_input.json", manifest)
        return manifest

    def fake_audio_publisher(
        final_wav: Path,
        output_path: Path,
        *,
        manifest_path: Path,
        overwrite: bool,
    ):
        nonlocal publication_calls
        publication_calls += 1
        assert overwrite is False
        assert manifest_path.name == "audio_publish_manifest.json"
        output_path.write_bytes(b"published audio")
        return {
            "status": "complete",
            "source_wav": str(final_wav),
            "output_audio": str(output_path),
        }

    def fake_runner(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        environment = kwargs["env"]  # type: ignore[assignment]
        assert str(Path(__file__).resolve().parents[1] / "src") in str(
            environment["PYTHONPATH"]  # type: ignore[index]
        )
        module = command[command.index("-m") + 1]
        if module in {"voicecut.streaming_narration", "voicecut.final_render"}:
            assert environment["GEMINI_API_KEY"] == "selected-secret"  # type: ignore[index]
        else:
            assert "GEMINI_API_KEY" not in environment
        assert "OPENAI_API_KEY" not in environment
        assert "DEEPSEEK_API_KEY" not in environment
        audio = args.work_dir / "00_media/source_audio.wav"
        audio_sha = sha256_file(audio)
        if module == "voicecut.analyze":
            output_dir = Path(command[command.index("--output-dir") + 1])
            write_json(output_dir / "analysis.json", {"audio_sha256": audio_sha})
        elif module == "voicecut.transcribe_mlx":
            write_json(
                Path(command[command.index("--output") + 1]),
                {"audio_sha256": audio_sha},
            )
        elif module == "voicecut.ctc_enrich":
            output_dir = Path(command[command.index("--output-dir") + 1])
            write_json(
                output_dir / "source_transcript_ctc_enriched.json",
                {
                    "audio_sha256": audio_sha,
                    "source_decode_strategy": (
                        "whisper_primary_plus_gated_raw_ctc_insertions_v1"
                    ),
                },
            )
        elif module == "voicecut.streaming_narration":
            output_dir = Path(command[command.index("--output-dir") + 1])
            transcript = Path(command[command.index("--transcript") + 1])
            write_json(
                output_dir / "streaming_plan.json",
                {
                    "status": "complete",
                    "backend": "gemini",
                    "model": "gemini-3.6-flash",
                    "transcript": str(transcript.resolve()),
                    "transcript_sha256": sha256_file(transcript),
                },
            )
        elif module == "voicecut.final_render":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            final_cut = output_dir / "final_cut.wav"
            final_cut.write_bytes(b"rendered audio")
            boundary_plan = output_dir / "final_boundary_plan.json"
            write_json(boundary_plan, {"status": "safe"})
            plan = Path(command[command.index("--plan") + 1])
            write_json(
                output_dir / "final_render_manifest.json",
                {
                    "status": "complete",
                    "renderer": "authoritative_single_pass_final_render_v2",
                    "source_audio_sha256": audio_sha,
                    "streaming_plan": str(plan.resolve()),
                    "streaming_plan_sha256": sha256_file(plan),
                    "effective_streaming_plan": str(plan.resolve()),
                    "effective_streaming_plan_sha256": sha256_file(plan),
                    "pause_planner_backend": "gemini",
                    "pause_planner_model": "gemini-3.6-flash",
                    "final_cut_wav": str(final_cut.resolve()),
                    "final_cut_wav_sha256": sha256_file(final_cut),
                    "final_boundary_plan": str(boundary_plan.resolve()),
                    "final_boundary_plan_sha256": sha256_file(boundary_plan),
                    "duration_seconds": 1.25,
                },
            )
        else:
            raise AssertionError(command)

    created = run_full_pipeline(
        args,
        runner=fake_runner,
        media_preparer=fake_media_preparer,
        audio_publisher=fake_audio_publisher,
        planner_preflight=_fake_gemini_preflight,
    )
    assert len(calls) == 5
    assert publication_calls == 1
    assert created["stages"] == {
        "media_input": "created",
        "analysis": "created",
        "transcription": "created",
        "hidden_retry_recovery": "created",
        "semantic_plan": "created",
        "final_render": "created",
        "publication": "created",
    }
    transcription_command = next(
        command for command in calls if "voicecut.transcribe_mlx" in command
    )
    assert transcription_command[0] == str(Path(sys.executable).absolute())
    assert created["output"] == str((tmp_path / "edited.mp3").absolute())

    def unexpected_runner(*_: object, **__: object) -> None:
        raise AssertionError("a cached rerun must not launch any stage")

    def unexpected_media(*_: object, **__: object):
        raise AssertionError("cached media must not be decoded")

    cached = run_full_pipeline(
        args,
        runner=unexpected_runner,
        media_preparer=unexpected_media,
        audio_publisher=unexpected_media,
        planner_preflight=_fake_gemini_preflight,
    )
    assert set(cached["stages"].values()) == {"cached"}
    assert cached["output_sha256"] == created["output_sha256"]


def test_public_parser_is_one_input_to_one_output(tmp_path: Path) -> None:
    source = tmp_path / "voice.m4a"
    parsed = build_parser().parse_args(
        [
            str(source),
            "-o",
            "edited.flac",
            "--planner-backend",
            "gemma",
            "--planner-model",
            "mlx-community/custom-gemma",
        ]
    )
    assert parsed.input == source
    assert parsed.output == Path("edited.flac")
    assert parsed.planner_backend == "gemma"
    assert parsed.planner_model == "mlx-community/custom-gemma"
    assert parsed.alignment_python == Path(sys.executable)
    assert parsed.debug_artifacts is False


def test_public_parser_can_enable_renderer_debug_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    parsed = build_parser().parse_args(
        [
            str(source),
            "--debug-artifacts",
        ]
    )
    assert parsed.debug_artifacts is True


def test_full_pipeline_rejects_symlink_output_before_any_stage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source media")
    output_link = tmp_path / "edited.wav"
    output_link.symlink_to(source)
    args = _args(tmp_path, source)
    args.output = output_link
    args.overwrite = True

    def unexpected_media(*_: object, **__: object):
        raise AssertionError("unsafe output must fail before media preparation")

    with pytest.raises(FullPipelineError, match="symbolic link"):
        run_full_pipeline(args, media_preparer=unexpected_media)


def test_full_pipeline_rejects_output_resolving_to_input_through_symlinked_parent(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "recording.wav"
    source.write_bytes(b"source media")
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(source_dir, target_is_directory=True)
    output = alias_dir / source.name
    assert not output.is_symlink()
    args = _args(tmp_path, source)
    args.output = output
    args.overwrite = True

    def unexpected_media(*_: object, **__: object):
        raise AssertionError("resolved input/output collision must fail immediately")

    with pytest.raises(FullPipelineError, match="must be different"):
        run_full_pipeline(args, media_preparer=unexpected_media)


def test_full_pipeline_never_resets_a_stage_containing_the_input(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    media_stage = work_dir / "00_media"
    media_stage.mkdir(parents=True)
    source = media_stage / "recording.wav"
    source.write_bytes(b"must survive")
    args = _args(tmp_path, source)

    with pytest.raises(FullPipelineError, match="input path.*disjoint"):
        run_full_pipeline(args)

    assert source.read_bytes() == b"must survive"


def test_full_pipeline_never_resets_a_stage_containing_the_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"source")
    work_dir = tmp_path / "work"
    publication_stage = work_dir / "06_publication"
    publication_stage.mkdir(parents=True)
    output = publication_stage / "edited.mp3"
    output.write_bytes(b"must survive")
    args = _args(tmp_path, source)
    args.output = output
    args.overwrite = True

    with pytest.raises(FullPipelineError, match="output path.*disjoint"):
        run_full_pipeline(args)

    assert output.read_bytes() == b"must survive"


def test_full_pipeline_rejects_work_dir_nested_under_future_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"source")
    output = tmp_path / "edited.mp3"
    args = _args(tmp_path, source)
    args.output = output
    args.work_dir = output / "cache"

    with pytest.raises(FullPipelineError, match="output path.*disjoint"):
        run_full_pipeline(args)

    assert not output.exists()


def test_full_pipeline_routes_video_input_to_video_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    args = build_parser().parse_args(
        [
            str(source),
            "--output",
            str(tmp_path / "edited.mp4"),
            "--work-dir",
            str(tmp_path / "work"),
            "--asr-python",
            sys.executable,
            "--alignment-python",
            sys.executable,
            "--planner-python",
            sys.executable,
            "--planner-backend",
            "gemini",
            "--planner-model",
            "gemini-3.6-flash",
        ]
    )
    calls: list[list[str]] = []
    video_calls = 0

    def fake_media_preparer(input_path: Path, output_dir: Path):
        output_dir.mkdir(parents=True)
        canonical = output_dir / "source_audio.wav"
        canonical.write_bytes(b"canonical audio")
        manifest = {
            "status": "complete",
            "source_media": str(input_path.resolve()),
            "source_media_sha256": sha256_file(input_path),
            "source_kind": "video",
            "canonical_audio": str(canonical.resolve()),
            "canonical_audio_sha256": sha256_file(canonical),
        }
        write_json(output_dir / "media_input.json", manifest)
        return manifest

    def fake_runner(command: list[str], **_: object) -> None:
        calls.append(command)
        module = command[command.index("-m") + 1]
        audio = args.work_dir / "00_media/source_audio.wav"
        audio_sha = sha256_file(audio)
        if module == "voicecut.analyze":
            output_dir = Path(command[command.index("--output-dir") + 1])
            write_json(output_dir / "analysis.json", {"audio_sha256": audio_sha})
        elif module == "voicecut.transcribe_mlx":
            write_json(
                Path(command[command.index("--output") + 1]),
                {"audio_sha256": audio_sha},
            )
        elif module == "voicecut.ctc_enrich":
            output_dir = Path(command[command.index("--output-dir") + 1])
            write_json(
                output_dir / "source_transcript_ctc_enriched.json",
                {
                    "audio_sha256": audio_sha,
                    "source_decode_strategy": (
                        "whisper_primary_plus_gated_raw_ctc_insertions_v1"
                    ),
                },
            )
        elif module == "voicecut.streaming_narration":
            output_dir = Path(command[command.index("--output-dir") + 1])
            transcript = Path(command[command.index("--transcript") + 1])
            write_json(
                output_dir / "streaming_plan.json",
                {
                    "status": "complete",
                    "backend": "gemini",
                    "model": "gemini-3.6-flash",
                    "transcript": str(transcript.resolve()),
                    "transcript_sha256": sha256_file(transcript),
                },
            )
        elif module == "voicecut.final_render":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            final_cut = output_dir / "final_cut.wav"
            final_cut.write_bytes(b"rendered audio")
            boundary_plan = output_dir / "final_boundary_plan.json"
            write_json(boundary_plan, {"status": "safe"})
            plan = Path(command[command.index("--plan") + 1])
            write_json(
                output_dir / "final_render_manifest.json",
                {
                    "status": "complete",
                    "renderer": "authoritative_single_pass_final_render_v2",
                    "source_audio_sha256": audio_sha,
                    "streaming_plan": str(plan.resolve()),
                    "streaming_plan_sha256": sha256_file(plan),
                    "effective_streaming_plan": str(plan.resolve()),
                    "effective_streaming_plan_sha256": sha256_file(plan),
                    "pause_planner_backend": "gemini",
                    "pause_planner_model": "gemini-3.6-flash",
                    "final_cut_wav": str(final_cut.resolve()),
                    "final_cut_wav_sha256": sha256_file(final_cut),
                    "final_boundary_plan": str(boundary_plan.resolve()),
                    "final_boundary_plan_sha256": sha256_file(boundary_plan),
                    "duration_seconds": 1.25,
                },
            )
        else:
            raise AssertionError(command)

    def unexpected_audio(*_: object, **__: object):
        raise AssertionError("video input must not use audio-only publication")

    def fake_video_publisher(
        *,
        source_video: Path,
        media_input_manifest_path: Path,
        final_render_manifest_path: Path,
        output_path: Path,
        manifest_path: Path,
        overwrite: bool,
    ):
        nonlocal video_calls
        video_calls += 1
        assert source_video == source.resolve()
        assert media_input_manifest_path.name == "media_input.json"
        assert final_render_manifest_path.name == "final_render_manifest.json"
        assert manifest_path.name == "video_render_manifest.json"
        assert overwrite is False
        output_path.write_bytes(b"published video")
        write_json(manifest_path, {"status": "complete"})
        return {
            "status": "complete",
            "source_video": str(source_video),
            "output_video": str(output_path),
        }

    result = run_full_pipeline(
        args,
        runner=fake_runner,
        media_preparer=fake_media_preparer,
        audio_publisher=unexpected_audio,
        video_publisher=fake_video_publisher,
        planner_preflight=_fake_gemini_preflight,
    )

    assert len(calls) == 5
    assert video_calls == 1
    assert result["input_kind"] == "video"
    assert result["output_kind"] == "video"
    assert Path(result["output"]).read_bytes() == b"published video"


def test_planner_preflight_fails_before_media_work(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source media")
    media_called = False

    def failed_preflight(**_: object) -> PlannerRuntimeConfiguration:
        raise RuntimeError("GEMINI_API_KEY is missing")

    def unexpected_media(*_: object, **__: object) -> dict[str, object]:
        nonlocal media_called
        media_called = True
        return {}

    with pytest.raises(
        FullPipelineError,
        match="planner preflight failed: GEMINI_API_KEY is missing",
    ):
        run_full_pipeline(
            _args(tmp_path, source),
            media_preparer=unexpected_media,
            planner_preflight=failed_preflight,
        )
    assert media_called is False


def test_effective_endpoint_is_canonicalized_in_cache_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source media")
    args = _args(tmp_path, source)
    args.planner_backend = "openai"
    args.planner_model = "gpt-5-mini"
    args.planner_base_url = "HTTPS://Planner.Example.TEST:443/v1/"

    class StopBeforeMedia(RuntimeError):
        pass

    def fake_preflight(**values: object) -> PlannerRuntimeConfiguration:
        assert values["base_url"] == "https://planner.example.test/v1"
        return PlannerRuntimeConfiguration(
            provider="openai",
            base_url="https://planner.example.test/v1",
            api_key_env="OPENAI_API_KEY",
        )

    def stop_media(*_: object, **__: object) -> dict[str, object]:
        raise StopBeforeMedia

    with pytest.raises(StopBeforeMedia):
        run_full_pipeline(
            args,
            media_preparer=stop_media,
            planner_preflight=fake_preflight,
        )
    configuration = read_json(args.work_dir / "pipeline_config.json")
    assert configuration["planner_base_url"] == "https://planner.example.test/v1"


def test_stage_failure_is_concise_and_endpoint_is_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint = "https://private.example.test/v1"

    def failed_runner(
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[object]:
        raise subprocess.CalledProcessError(
            23,
            command,
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "worker.py", line 10, in <module>\n'
                f"RuntimeError: endpoint {endpoint} refused the request\n"
            ),
        )

    with pytest.raises(
        FullPipelineError,
        match="TEST STAGE failed with exit status 23",
    ) as raised:
        _run(
            "TEST STAGE",
            [
                sys.executable,
                "-m",
                "voicecut.streaming_narration",
                "--planner-base-url",
                endpoint,
            ],
            runner=failed_runner,
        )
    assert endpoint not in str(raised.value)
    assert "RuntimeError: endpoint <redacted-endpoint> refused the request" in str(
        raised.value
    )
    captured = capsys.readouterr()
    assert endpoint not in captured.out
    assert "<redacted-endpoint>" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_planner_environment_retains_only_selected_standard_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "selected")
    monkeypatch.setenv("GEMINI_API_KEY", "unrelated")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unrelated")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://endpoint.example.test/v1")

    planner_environment = _subprocess_environment(planner_backend="openai")
    assert planner_environment["OPENAI_API_KEY"] == "selected"
    assert "GEMINI_API_KEY" not in planner_environment
    assert "DEEPSEEK_API_KEY" not in planner_environment
    assert "OPENAI_BASE_URL" not in planner_environment

    analysis_environment = _subprocess_environment()
    assert "OPENAI_API_KEY" not in analysis_environment


def test_implementation_fingerprint_is_content_addressed() -> None:
    fingerprint = _implementation_fingerprint()
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0


def test_work_directory_lock_rejects_a_concurrent_owner(tmp_path: Path) -> None:
    work_dir = tmp_path / "shared-cache"
    first = _WorkDirectoryLock(work_dir)
    try:
        with pytest.raises(FullPipelineError, match="another VoiceCut run owns"):
            _WorkDirectoryLock(work_dir)
    finally:
        first.close()

    second = _WorkDirectoryLock(work_dir)
    second.close()
