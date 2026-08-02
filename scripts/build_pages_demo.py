from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_SOURCE = PROJECT_ROOT / "docs" / "demo"
MEDIA_SOURCE = PROJECT_ROOT / "examples" / "media"
README_ASSETS = PROJECT_ROOT / "docs" / "assets" / "readme"
REQUIRED_ASSETS = (
    "pipeline.svg",
    "video-before.jpg",
    "video-after.jpg",
)


def _safe_output_path(value: str) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    allowed_root = (PROJECT_ROOT / ".pages-site").resolve()
    if output != allowed_root and allowed_root not in output.parents:
        raise ValueError("demo output must be .pages-site or one of its subdirectories")
    return output


def build_demo(output: Path) -> None:
    required_media = (
        MEDIA_SOURCE / "audio" / "example_en.wav",
        MEDIA_SOURCE / "audio" / "example_en_edited.wav",
        MEDIA_SOURCE / "audio" / "example_ru.wav",
        MEDIA_SOURCE / "audio" / "example_ru_edited.wav",
        MEDIA_SOURCE / "video" / "video.mp4",
        MEDIA_SOURCE / "video" / "video_edited.mp4",
    )
    required_files = (
        SITE_SOURCE / "index.html",
        SITE_SOURCE / "styles.css",
        SITE_SOURCE / "player.js",
        *(README_ASSETS / name for name in REQUIRED_ASSETS),
        *required_media,
    )
    missing = [
        path.relative_to(PROJECT_ROOT) for path in required_files if not path.is_file()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"cannot build the demo; missing: {joined}")

    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(SITE_SOURCE, output)
    for source in required_media:
        destination = output / "media" / source.relative_to(MEDIA_SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assets_output = output / "assets"
    assets_output.mkdir()
    for name in REQUIRED_ASSETS:
        shutil.copy2(README_ASSETS / name, assets_output / name)
    (output / ".nojekyll").write_text("", encoding="utf-8")

    relative_output = output.relative_to(PROJECT_ROOT)
    print(f"GitHub Pages demo built at {relative_output}/index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static VoiceCut demo site")
    parser.add_argument(
        "--output",
        default=".pages-site",
        help="repository-local output directory (default: .pages-site)",
    )
    args = parser.parse_args()
    build_demo(_safe_output_path(args.output))


if __name__ == "__main__":
    main()
