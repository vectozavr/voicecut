# VoiceCut

VoiceCut turns a raw, retake-heavy scripted narration recording into a cut,
breath-cleaned, mastered WAV. It is built for voice-over recordings containing
false starts, repeated takes, abandoned clauses, long pauses, and breaths that
ordinary silence removers cannot handle safely.

VoiceCut keeps your original voice. It does not synthesize or clone speech, and
all processing runs locally without an OpenAI, Gemini, or other cloud API key.

## How it works

VoiceCut uses the script to decide **what** should remain and the waveform to
decide **where** edits are safe:

1. Silero VAD and waveform features divide the recording into non-overlapping
   speech regions and find clean room tone.
2. MLX Whisper transcribes the full recording and each region independently so
   retries remain visible.
3. A global, monotonic alignment selects the best recorded take for every
   script phrase and penalizes repetitions, fillers, missing words, and
   suspicious pauses.
4. WhisperX forced alignment anchors retained words; quiet waveform valleys and
   zero crossings determine the final sample-level cuts.
5. VoiceCut inserts punctuation-aware room-tone pauses, applies smooth
   equal-power crossfades, and attenuates only confidently detected breath
   cores between protected word boundaries.
6. FFmpeg masters the result to 48 kHz, mono, 24-bit PCM at approximately
   -16 LUFS and no more than -1.5 dBTP.
7. The edited audio is transcribed again by two recognizers and checked for
   missing content, extra repetitions, unsafe boundaries, clicks, dropouts,
   clipping, breath-edit safety, and loudness.

The quality gate is fail-closed: `production.wav` is created only when the
enabled checks pass. Ambiguous runs keep `candidate.wav` and produce reports
and short review clips.

## Requirements

- macOS on Apple Silicon for the default MLX Whisper recognizer
- Python 3.11
- `ffmpeg` and `ffprobe` on `PATH`
- several gigabytes of disk space for local speech models

Install FFmpeg with Homebrew:

```bash
brew install ffmpeg
```

VoiceCut uses separate environments for the audio stack and MLX:

```bash
git clone https://github.com/vectozavr/voicecut.git
cd voicecut

python3.11 -m venv .venv-audio
.venv-audio/bin/pip install -e ".[audio]"

python3.11 -m venv .venv-mlx
.venv-mlx/bin/pip install -e ".[mlx]"
```

The first run downloads Whisper, WhisperX alignment, Silero VAD, and
Faster-Whisper models.

## Quick start

Provide a raw recording and the authoritative script in spoken order:

```bash
.venv-audio/bin/voicecut \
  --audio raw_narration.wav \
  --script narration_script.md \
  --output-dir narration_run
```

The output directory must be new or empty. Resume an interrupted run with:

```bash
.venv-audio/bin/voicecut \
  --audio raw_narration.wav \
  --script narration_script.md \
  --output-dir narration_run \
  --resume
```

If the environments are elsewhere, pass `--audio-python` and `--mlx-python`.

## Preparing the recording and script

- The script should describe the final narration you want, not necessarily the
  first draft you wrote.
- If you intentionally paraphrased while recording, update the script to the
  wording you want VoiceCut to retain.
- Delete intentionally skipped sentences from the script, or explicitly waive
  a parsed sentence with `--allow-missing-unit N`.
- Leave a little clean room tone in the recording.
- Markdown headings, bracketed stage cues, fenced code blocks, standalone URLs,
  and common resource lines such as `Code: [project](https://...)` are ignored.

For project names or recurring ASR mistakes, create an alias file:

```json
{
  "aliases": {
    "my misheard name": "my project name",
    "misrecognized acronym": "correct acronym"
  }
}
```

Then add `--aliases aliases.json`. See
[`examples/aliases.json`](examples/aliases.json).

## Important options

```text
--allow-missing-unit N       Explicitly waive an unrecorded parsed sentence
--aliases FILE               Normalize names and recurring ASR mistakes
--resume                     Reuse verified checkpoints
--skip-independent-qa        Skip the slower second ASR opinion
--skip-ctc                   Skip forced alignment; weakens boundary evidence
--disable-breath-attenuation Preserve all internal breaths
--target-lufs -16            Set the mastering loudness target
--target-peak -1.5           Set the true-peak ceiling
--accept-review              Publish REVIEW after manual inspection, never FAIL
--no-strict                  Return zero on REVIEW/FAIL without publishing it
```

Run `voicecut --help` for the complete list.

## Outputs

```text
narration_run/
  production.wav            Published only on PASS or accepted REVIEW
  candidate.wav             Diagnostic result when strict QA does not pass
  edit_plan.json            Selected takes, alternatives, and review reasons
  edit_decision_list.json   Exact source/output sample mapping
  mastering_report.json     Measured loudness and output format
  qa_report.json            PASS, REVIEW, or FAIL with supporting evidence
  pipeline_result.json      Small machine-readable result
  run_manifest.json         Input hashes, configuration, and tool versions
  review_clips/             Short clips around reported findings
  logs/                     Complete stage logs
  work/                     Resumable intermediate artifacts
```

Automation should read `pipeline_result.json` rather than infer success only
from the process exit code.

## Limits

VoiceCut cannot invent an unrecorded sentence or reliably decide whether a
different spoken meaning was intentional. It also avoids cutting through
continuously voiced audio when no safe boundary exists. Those cases are
reported as `REVIEW` or `FAIL` instead of being silently accepted.

Always audition the beginning, ending, technical names, formulas, and reported
review clips before publishing important narration.

## Development

Run the regression suite from the repository root:

```bash
PYTHONPATH=src .venv-audio/bin/python -m unittest discover -s tests -v
```

## License

VoiceCut is released under the [MIT License](LICENSE).
