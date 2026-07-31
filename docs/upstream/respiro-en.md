# Respiro-en provenance

VoiceCut uses the official Respiro-en implementation for optional breath
evidence. Normal installation downloads these three files from an immutable
upstream revision into VoiceCut's runtime-model cache and verifies every file
before it can be loaded. The model checkpoint is not committed to this
repository.

- Repository: <https://github.com/ydqmkkx/Respiro-en>
- Pinned commit: `70e01c60c2f582c41092730680f2894ab24d6467`
- License: MIT, Copyright (c) 2024 DongYANG
- Paper: DongYANG, Tomoki Koriyama, Yuki Saito, “Frame-Wise Breath Detection
  with Self-Training: An Exploration of Enhancing Breath Naturalness in
  Text-to-Speech,” Interspeech 2024, pp. 4928–4932,
  <https://doi.org/10.21437/Interspeech.2024-168>

## Verified upstream files

| File | SHA-256 |
| --- | --- |
| `modules.py` | `f789e0986e3090d7df5f9f0f596d9e3601c6da514c3ac01a65920a493b840e46` |
| `respiro-en.pt` | `1f4a9b96f96645c480bf0e07b1e18cd68878ac0b4bb5dc920ad93f9b17df858a` |
| `LICENSE` | `a34ad1af58dc7c02f867f620f7ddc952029b383c9b0dce349d54f6b875e079cd` |

The installer downloads from URLs rooted at:

```text
https://raw.githubusercontent.com/ydqmkkx/Respiro-en/70e01c60c2f582c41092730680f2894ab24d6467/
```

It aborts when an existing or newly downloaded file does not match the pinned
hash. Runtime code repeats this verification before importing the upstream
model definition or loading the checkpoint.

## Upstream runtime notes

The official example chooses CUDA when available and CPU otherwise. Its
checkpoint contains CUDA-tagged tensors, so the example as published does not
load unchanged on a CPU-only host. VoiceCut uses PyTorch's documented
`map_location` and weights-only loading options; the upstream `DetectionNet`
implementation itself remains unchanged.

Respiro-en produces one probability every 10 ms. The upstream helper's
`min_length` comment and implementation use different units, so VoiceCut does
not reuse that thresholding helper: its CLI duration is converted explicitly
from milliseconds to 10 ms frames.
