from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from voicecut.forced_align_worker import run_alignment_jobs
from voicecut.language_profiles import get_language_profile


def test_russian_worker_loads_the_pinned_whisperx_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = get_language_profile("ru")
    crop = tmp_path / "context.wav"
    crop.write_bytes(b"mock crop")
    jobs_path = tmp_path / "jobs.json"
    output_path = tmp_path / "alignment.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "clip_index": 4,
                        "crop_wav": str(crop),
                        "crop_duration_seconds": 1.0,
                        "local_source_text": "это русский пример",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    load_calls: list[dict[str, object]] = []
    align_calls: list[dict[str, object]] = []
    fake = types.ModuleType("whisperx")

    def load_align_model(**kwargs):
        load_calls.append(kwargs)
        return object(), {"language": "ru", "type": "huggingface"}

    def load_audio(_: str):
        return [0.0] * 16_000

    def align(segments, model, metadata, audio, device, **kwargs):
        del model, metadata, audio
        align_calls.append(
            {
                "segments": segments,
                "device": device,
                **kwargs,
            }
        )
        return {"segments": segments, "word_segments": []}

    fake.load_align_model = load_align_model
    fake.load_audio = load_audio
    fake.align = align
    monkeypatch.setitem(sys.modules, "whisperx", fake)

    result = run_alignment_jobs(
        jobs_path=jobs_path,
        output_path=output_path,
        language="ru",
        align_model=profile.whisperx_model,
        device="cpu",
    )

    assert load_calls == [
        {
            "language_code": "ru",
            "device": "cpu",
            "model_name": profile.whisperx_model,
        }
    ]
    assert align_calls[0]["segments"][0]["text"] == "это русский пример"
    assert align_calls[0]["return_char_alignments"] is True
    assert result["language"] == "ru"
    assert result["requested_model"] == profile.whisperx_model
    assert result["jobs"][0]["error"] is None
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
