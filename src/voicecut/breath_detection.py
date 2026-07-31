"""Pinned Respiro-en frame probabilities mapped to canonical source samples."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

from .common import sha256_file


RESPIRO_UPSTREAM_REPOSITORY = "https://github.com/ydqmkkx/Respiro-en"
RESPIRO_UPSTREAM_COMMIT = "70e01c60c2f582c41092730680f2894ab24d6467"
RESPIRO_MODULES_SHA256 = (
    "f789e0986e3090d7df5f9f0f596d9e3601c6da514c3ac01a65920a493b840e46"
)
RESPIRO_CHECKPOINT_SHA256 = (
    "1f4a9b96f96645c480bf0e07b1e18cd68878ac0b4bb5dc920ad93f9b17df858a"
)
RESPIRO_LICENSE_SHA256 = (
    "a34ad1af58dc7c02f867f620f7ddc952029b383c9b0dce349d54f6b875e079cd"
)
RESPIRO_SAMPLE_RATE = 16_000
RESPIRO_FRAME_HOP_SAMPLES = 160
RESPIRO_FRAME_HOP_MS = 10
DEFAULT_BREATH_THRESHOLD = 0.5
DEFAULT_BREATH_MIN_DURATION_MS = 80
DEFAULT_BREATH_CONTEXT_MS = 750.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPIRO_CACHE_ROOT = (
    PROJECT_ROOT
    / ".voicecut-cache"
    / "runtime"
    / "respiro-en"
    / RESPIRO_UPSTREAM_COMMIT
)


class BreathDetectionError(RuntimeError):
    """Pinned Respiro-en evidence could not be produced safely."""


@dataclass(frozen=True)
class RespiroRuntime:
    """Verified upstream model plus its unchanged feature extractor."""

    model: Any
    feature_extractor: Callable[..., tuple[Any, Any]]
    device: Any
    device_name: str
    cache_root: Path

    def infer(self, waveform_16khz: np.ndarray) -> np.ndarray:
        import torch

        waveform = np.asarray(waveform_16khz, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform):
            raise BreathDetectionError("Respiro-en requires a non-empty mono crop")
        feature, length = self.feature_extractor(
            waveform,
            sr=RESPIRO_SAMPLE_RATE,
        )
        feature = feature.to(self.device)
        length = length.to(self.device)
        with torch.inference_mode():
            output = self.model(feature, length)
        probabilities = output[0].detach().to("cpu").numpy().astype(np.float32)
        return _validate_probabilities(probabilities)


_RUNTIME_CACHE: dict[tuple[Path, str], RespiroRuntime] = {}


def _verified_runtime_file(
    cache_root: Path,
    filename: str,
    expected_sha256: str,
) -> Path:
    path = (cache_root / filename).absolute()
    if not path.is_file() or path.is_symlink():
        raise BreathDetectionError(
            f"pinned Respiro-en file is missing or unsafe: {path}; "
            "run scripts/install.sh"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise BreathDetectionError(
            f"pinned Respiro-en hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return path


def verify_respiro_runtime(
    cache_root: Path = DEFAULT_RESPIRO_CACHE_ROOT,
) -> dict[str, str]:
    """Verify every pinned upstream file before importing executable code."""

    root = cache_root.expanduser().absolute()
    modules_path = _verified_runtime_file(
        root,
        "modules.py",
        RESPIRO_MODULES_SHA256,
    )
    checkpoint_path = _verified_runtime_file(
        root,
        "respiro-en.pt",
        RESPIRO_CHECKPOINT_SHA256,
    )
    license_path = _verified_runtime_file(
        root,
        "LICENSE",
        RESPIRO_LICENSE_SHA256,
    )
    return {
        "cache_root": str(root),
        "modules_path": str(modules_path),
        "checkpoint_path": str(checkpoint_path),
        "license_path": str(license_path),
    }


def _import_upstream_modules(path: Path) -> ModuleType:
    module_name = f"voicecut_respiro_en_{RESPIRO_UPSTREAM_COMMIT}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BreathDetectionError(f"cannot import pinned Respiro-en module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise BreathDetectionError(
            f"pinned Respiro-en module import failed: {type(error).__name__}: {error}"
        ) from error
    return module


def load_respiro_runtime(
    *,
    cache_root: Path = DEFAULT_RESPIRO_CACHE_ROOT,
    device: str | None = None,
) -> RespiroRuntime:
    """Load the verified official DetectionNet without modifying its source."""

    import torch

    paths = verify_respiro_runtime(cache_root)
    root = Path(paths["cache_root"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (root, device)
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached

    upstream = _import_upstream_modules(Path(paths["modules_path"]))
    detection_net = getattr(upstream, "DetectionNet", None)
    feature_extractor = getattr(upstream, "feature_extractor", None)
    if detection_net is None or not callable(feature_extractor):
        raise BreathDetectionError(
            "pinned Respiro-en module has no DetectionNet/feature_extractor"
        )
    torch_device = torch.device(device)
    try:
        model = detection_net().to(torch_device)
        checkpoint = torch.load(
            paths["checkpoint_path"],
            map_location=torch_device,
            weights_only=True,
        )
        state_dict = checkpoint["model"]
        load_result = model.load_state_dict(state_dict)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise BreathDetectionError(
                "Respiro-en checkpoint has incompatible state-dict keys"
            )
        model.eval()
    except BreathDetectionError:
        raise
    except Exception as error:
        raise BreathDetectionError(
            f"Respiro-en model load failed: {type(error).__name__}: {error}"
        ) from error

    runtime = RespiroRuntime(
        model=model,
        feature_extractor=feature_extractor,
        device=torch_device,
        device_name=str(torch_device),
        cache_root=root,
    )
    _RUNTIME_CACHE[key] = runtime
    return runtime


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise BreathDetectionError("Respiro-en returned no frame probabilities")
    if not np.all(np.isfinite(values)):
        raise BreathDetectionError("Respiro-en returned non-finite probabilities")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise BreathDetectionError("Respiro-en probabilities leave [0, 1]")
    return values


def merge_analysis_crops(
    ranges: Sequence[tuple[int, int]],
    *,
    total_samples: int,
    sample_rate: int,
    context_ms: float = DEFAULT_BREATH_CONTEXT_MS,
) -> list[tuple[int, int]]:
    """Expand relevant regions once and merge overlaps before inference."""

    if total_samples < 0 or sample_rate <= 0 or context_ms < 0.0:
        raise ValueError("invalid breath-analysis crop geometry")
    context = round(context_ms * sample_rate / 1000.0)
    expanded: list[tuple[int, int]] = []
    for raw_start, raw_end in ranges:
        start = max(0, int(raw_start) - context)
        end = min(total_samples, int(raw_end) + context)
        if not 0 <= start < end <= total_samples:
            continue
        expanded.append((start, end))
    merged: list[list[int]] = []
    for start, end in sorted(expanded):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _resample_mono(
    mono: np.ndarray,
    *,
    source_sample_rate: int,
) -> np.ndarray:
    waveform = np.asarray(mono, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform):
        raise BreathDetectionError("breath-analysis crop is empty")
    if source_sample_rate == RESPIRO_SAMPLE_RATE:
        return np.array(waveform, dtype=np.float32, copy=True)

    import torch
    import torchaudio.functional as audio_functional

    tensor = torch.from_numpy(np.ascontiguousarray(waveform))
    resampled = audio_functional.resample(
        tensor,
        source_sample_rate,
        RESPIRO_SAMPLE_RATE,
    )
    return resampled.detach().to("cpu").numpy().astype(np.float32, copy=False)


def _event_frame_runs(
    probabilities: np.ndarray,
    *,
    threshold: float,
    minimum_duration_ms: int,
) -> list[tuple[int, int]]:
    minimum_frames = max(
        1,
        math.ceil(minimum_duration_ms / RESPIRO_FRAME_HOP_MS),
    )
    indices = np.flatnonzero(probabilities >= threshold)
    if not len(indices):
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    return [
        (int(run[0]), int(run[-1]) + 1)
        for run in np.split(indices, split_points)
        if len(run) >= minimum_frames
    ]


def _frame_to_canonical_offset(
    frame_index: int,
    *,
    source_sample_rate: int,
    round_up: bool,
) -> int:
    numerator = frame_index * source_sample_rate
    denominator = 1000 // RESPIRO_FRAME_HOP_MS
    if round_up:
        return (numerator + denominator - 1) // denominator
    return numerator // denominator


def _events_for_crop(
    *,
    probabilities: np.ndarray,
    crop_start_sample: int,
    crop_end_sample: int,
    source_sample_rate: int,
    threshold: float,
    minimum_duration_ms: int,
    crop_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for first_frame, end_frame in _event_frame_runs(
        probabilities,
        threshold=threshold,
        minimum_duration_ms=minimum_duration_ms,
    ):
        start = crop_start_sample + _frame_to_canonical_offset(
            first_frame,
            source_sample_rate=source_sample_rate,
            round_up=False,
        )
        end = crop_start_sample + _frame_to_canonical_offset(
            end_frame,
            source_sample_rate=source_sample_rate,
            round_up=True,
        )
        start = max(crop_start_sample, min(crop_end_sample, start))
        end = max(start, min(crop_end_sample, end))
        if end <= start:
            continue
        selected = probabilities[first_frame:end_frame]
        events.append(
            {
                "crop_id": crop_id,
                "first_frame": first_frame,
                "end_frame": end_frame,
                "start_seconds": start / source_sample_rate,
                "end_seconds": end / source_sample_rate,
                "start_sample": start,
                "end_sample": end,
                "maximum_probability": float(np.max(selected)),
                "mean_probability": float(np.mean(selected)),
            }
        )
    return events


def analyze_breath_evidence(
    *,
    source_audio: np.ndarray,
    source_sample_rate: int,
    relevant_ranges: Sequence[tuple[int, int]],
    threshold: float = DEFAULT_BREATH_THRESHOLD,
    minimum_duration_ms: int = DEFAULT_BREATH_MIN_DURATION_MS,
    context_ms: float = DEFAULT_BREATH_CONTEXT_MS,
    cache_root: Path = DEFAULT_RESPIRO_CACHE_ROOT,
    runtime: RespiroRuntime | None = None,
    probability_provider: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Analyze only expanded relevant crops and retain every frame probability."""

    audio = np.asarray(source_audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not len(audio):
        raise ValueError("source_audio must contain canonical audio frames")
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("breath threshold must be inside [0, 1]")
    if minimum_duration_ms <= 0:
        raise ValueError("breath minimum duration must be positive")

    crops = merge_analysis_crops(
        relevant_ranges,
        total_samples=len(audio),
        sample_rate=source_sample_rate,
        context_ms=context_ms,
    )
    base: dict[str, Any] = {
        "schema_version": 1,
        "backend": "respiro-en",
        "upstream_repository": RESPIRO_UPSTREAM_REPOSITORY,
        "upstream_commit": RESPIRO_UPSTREAM_COMMIT,
        "modules_sha256": RESPIRO_MODULES_SHA256,
        "checkpoint_sha256": RESPIRO_CHECKPOINT_SHA256,
        "license_sha256": RESPIRO_LICENSE_SHA256,
        "sample_rate": RESPIRO_SAMPLE_RATE,
        "frame_hop_ms": RESPIRO_FRAME_HOP_MS,
        "threshold": threshold,
        "minimum_duration_ms": minimum_duration_ms,
        "context_ms": context_ms,
        "analysis_crops": [],
        "events": [],
    }
    if not crops:
        return {
            **base,
            "status": "no_relevant_regions",
            "execution_device": None,
        }

    if probability_provider is None:
        active_runtime = runtime or load_respiro_runtime(cache_root=cache_root)
        probability_provider = active_runtime.infer
        execution_device = active_runtime.device_name
    else:
        execution_device = "injected_probability_provider"

    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    crop_records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for crop_index, (crop_start, crop_end) in enumerate(crops):
        crop_id = f"breath_crop_{crop_index:04d}"
        waveform_16khz = _resample_mono(
            mono[crop_start:crop_end],
            source_sample_rate=source_sample_rate,
        )
        probabilities = _validate_probabilities(probability_provider(waveform_16khz))
        expected_frames = len(waveform_16khz) // RESPIRO_FRAME_HOP_SAMPLES + 1
        if len(probabilities) != expected_frames:
            raise BreathDetectionError(
                f"Respiro-en returned {len(probabilities)} frames for "
                f"{len(waveform_16khz)} inference samples; expected "
                f"{expected_frames}"
            )
        crop_records.append(
            {
                "crop_id": crop_id,
                "source_start_sample": crop_start,
                "source_end_sample": crop_end,
                "source_start_seconds": crop_start / source_sample_rate,
                "source_end_seconds": crop_end / source_sample_rate,
                "inference_sample_count": len(waveform_16khz),
                "frame_count": len(probabilities),
                "frame_probabilities": [float(value) for value in probabilities],
            }
        )
        events.extend(
            _events_for_crop(
                probabilities=probabilities,
                crop_start_sample=crop_start,
                crop_end_sample=crop_end,
                source_sample_rate=source_sample_rate,
                threshold=threshold,
                minimum_duration_ms=minimum_duration_ms,
                crop_id=crop_id,
            )
        )

    return {
        **base,
        "status": "complete",
        "execution_device": execution_device,
        "analysis_crops": crop_records,
        "events": sorted(events, key=lambda item: int(item["start_sample"])),
    }


__all__ = [
    "BreathDetectionError",
    "DEFAULT_BREATH_CONTEXT_MS",
    "DEFAULT_BREATH_MIN_DURATION_MS",
    "DEFAULT_BREATH_THRESHOLD",
    "DEFAULT_RESPIRO_CACHE_ROOT",
    "RESPIRO_CHECKPOINT_SHA256",
    "RESPIRO_FRAME_HOP_MS",
    "RESPIRO_LICENSE_SHA256",
    "RESPIRO_MODULES_SHA256",
    "RESPIRO_SAMPLE_RATE",
    "RESPIRO_UPSTREAM_COMMIT",
    "RespiroRuntime",
    "analyze_breath_evidence",
    "load_respiro_runtime",
    "merge_analysis_crops",
    "verify_respiro_runtime",
]
