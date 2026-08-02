# Credits and third-party software

VoiceCut is an independent open-source project released under the
[MIT License](../LICENSE). It relies on research, software, and models created
by other communities. Their names and trademarks remain the property of their
respective owners.

The projects listed here do not sponsor, endorse, certify, or provide support
for VoiceCut. Likewise, VoiceCut's MIT license does not replace the licenses or
terms that apply to third-party software, model weights, media codecs, or a
planner selected by the user.

Pinned package versions are recorded in [`pyproject.toml`](../pyproject.toml)
and [`environment-mfa.yml`](../environment-mfa.yml). Those files, rather than
this narrative overview, are the authoritative dependency manifest.

## Speech analysis and alignment

### Montreal Forced Aligner

[Montreal Forced Aligner (MFA)](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
is the sole source of production word- and phone-level cut coordinates in
VoiceCut. VoiceCut invokes the MFA 3.4.1 command-line interface in an isolated
micromamba environment, resolves every boundary before rendering, and then
slices the canonical source once. Whisper and waveform timestamps are not
fallback cut coordinates.

- Software license: [MIT](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/LICENSE)
- Documentation: [MFA 3.4.1 user guide](https://montreal-forced-aligner.readthedocs.io/en/v3.4.1/user_guide/)
- English model: [`english_us_arpa`](https://huggingface.co/MontrealCorpusTools/english_us_arpa), CC BY 4.0
- Russian model: [`russian_mfa` at VoiceCut's pinned revision](https://huggingface.co/MontrealCorpusTools/russian_mfa/tree/88b81ae3eaf3bd8163bb3f7c43e1ae61478595af), CC BY 4.0
- Citation: Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner,
  and Morgan Sonderegger, “Montreal Forced Aligner: Trainable Text-Speech
  Alignment Using Kaldi,” Interspeech 2017,
  [doi:10.21437/Interspeech.2017-1386](https://doi.org/10.21437/Interspeech.2017-1386)

The MFA model cards describe their own intended uses, limitations, training
data, and citations. Those model terms apply separately from the MFA software
license and the VoiceCut license.

### WhisperX

[WhisperX](https://github.com/m-bain/whisperX) is used locally only for the
retained-occurrence completeness veto. Its character coverage and edge scores
can reject an incomplete or weak take before boundary planning. WhisperX does
**not** supply final production cut coordinates; MFA remains authoritative for
those coordinates.

- Pinned package version: 3.8.6
- License: [BSD 2-Clause](https://github.com/m-bain/whisperX/blob/main/LICENSE)
- Citation: Max Bain, Jaesung Huh, Tengda Han, and Andrew Zisserman,
  “WhisperX: Time-Accurate Speech Transcription of Long-Form Audio,”
  Interspeech 2023,
  [doi:10.21437/Interspeech.2023-78](https://doi.org/10.21437/Interspeech.2023-78)

### Silero VAD

[Silero VAD](https://github.com/snakers4/silero-vad) supplies voice-activity
evidence used to divide recordings into manageable analysis regions. It does
not choose semantic content or override MFA-protected speech boundaries.

- Pinned package version: 6.2.1
- License: [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE)
- Upstream citation: Silero Team, “Silero VAD: pre-trained enterprise-grade
  Voice Activity Detector,”
  [GitHub repository](https://github.com/snakers4/silero-vad)

## Transcription and Apple Silicon inference

### MLX and MLX Whisper

[MLX](https://github.com/ml-explore/mlx) provides local machine-learning
inference optimized for Apple silicon. VoiceCut uses
[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) to
produce the primary transcript and approximate word anchors. Those anchors
help locate alignment contexts, but MFA supplies the final phone coordinates.
VoiceCut also uses [MLX LM](https://github.com/ml-explore/mlx-lm) when a user
selects an optional local planner.

- Pinned packages: MLX 0.32.0, MLX Whisper 0.4.3, and MLX LM 0.31.3
- MLX license: [MIT](https://github.com/ml-explore/mlx/blob/main/LICENSE)
- MLX Examples/Whisper license:
  [MIT](https://github.com/ml-explore/mlx-examples/blob/main/LICENSE)
- Citation: Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan
  Collobert, “MLX: Efficient and flexible machine learning on Apple silicon,”
  [MLX software project](https://github.com/ml-explore)

MLX Whisper runs converted
[OpenAI Whisper](https://github.com/openai/whisper) checkpoints. Each selected
checkpoint or local planner model may carry its own model card, license, and
acceptable-use terms; users and distributors are responsible for reviewing
them. OpenAI Whisper's code and released model weights are distributed under
its [MIT license](https://github.com/openai/whisper/blob/main/LICENSE).

## Optional breath evidence

### Respiro-en

[Respiro-en](https://github.com/ydqmkkx/Respiro-en) provides optional
frame-wise English breath probabilities. VoiceCut uses those probabilities
only to plan duration-preserving replacement inside MFA-confirmed non-speech
and to screen room-tone candidates. Respiro-en cannot classify a sample as
non-speech, move an MFA endpoint, or authorize modification of a retained
phone. It is not validated for Russian, so Russian processing defaults to
breath cleanup off.

VoiceCut downloads `modules.py`, `respiro-en.pt`, and the upstream license from
one immutable commit and verifies all three hashes before use. The checkpoint
is not stored in this repository. Exact provenance and hashes are documented
in [`docs/upstream/respiro-en.md`](upstream/respiro-en.md).

- Pinned upstream commit: `70e01c60c2f582c41092730680f2894ab24d6467`
- License: [MIT at the pinned revision](https://github.com/ydqmkkx/Respiro-en/blob/70e01c60c2f582c41092730680f2894ab24d6467/LICENSE)
- Citation: Dong Yang, Tomoki Koriyama, and Yuki Saito, “Frame-Wise Breath
  Detection with Self-Training: An Exploration of Enhancing Breath Naturalness
  in Text-to-Speech,” Interspeech 2024,
  [doi:10.21437/Interspeech.2024-168](https://doi.org/10.21437/Interspeech.2024-168)

## Audio, media, and tensor runtimes

### PyTorch and TorchAudio

[PyTorch](https://github.com/pytorch/pytorch) executes the local neural audio
models, and [TorchAudio](https://github.com/pytorch/audio) provides audio
tensor and resampling operations used by those model adapters. They are
runtime infrastructure, not semantic editors or production boundary
authorities.

- Pinned packages: PyTorch 2.8.0 and TorchAudio 2.8.0
- PyTorch license: [BSD-style terms](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- TorchAudio license: [BSD 2-Clause](https://github.com/pytorch/audio/blob/main/LICENSE)
- TorchAudio citation: Yao-Yuan Yang et al., “TorchAudio: Building Blocks for
  Audio and Speech Processing,”
  [arXiv:2110.15018](https://arxiv.org/abs/2110.15018)

### FFmpeg

[FFmpeg](https://ffmpeg.org/) and FFprobe handle media inspection, canonical
audio extraction, format conversion, final audio encoding, and video
publication. VoiceCut invokes a separately installed FFmpeg distribution and
does not vendor an FFmpeg binary.

Most upstream FFmpeg source is LGPL 2.1 or later, while optional components can
make a particular build GPL or subject to other compatible licenses. Consult
the [official FFmpeg license overview](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)
and the configuration of the binary you install or redistribute.

## Additional libraries and services

VoiceCut also depends on projects including
[NumPy](https://numpy.org/), [SciPy](https://scipy.org/),
[SoundFile](https://github.com/bastibe/python-soundfile),
[librosa](https://librosa.org/),
[IntervalTree](https://github.com/chaimleib/intervaltree), and
[Matplotlib](https://matplotlib.org/). See `pyproject.toml` and the installed
package metadata for the exact versions and license texts.

Cloud planner SDKs are optional. A selected API provider processes transcript
text under its own service terms. Local and downloaded planner checkpoints
likewise retain their own model licenses. Using one of these integrations does
not imply that its provider endorses VoiceCut.

Thank you to all upstream authors, maintainers, researchers, model creators,
dataset contributors, and open-source communities whose work makes VoiceCut
possible.
