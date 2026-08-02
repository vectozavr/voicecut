from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

GITHUB_API_VERSION = "2026-03-10"


def _read_json_url(url: str, token: str | None = None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "voicecut-star-history",
    }
    if token:
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            }
        )
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _repository_snapshot(repository: str, token: str) -> tuple[date, int]:
    encoded_repository = urllib.parse.quote(repository, safe="/")
    metadata = _read_json_url(
        f"https://api.github.com/repos/{encoded_repository}", token
    )
    created_at = datetime.fromisoformat(metadata["created_at"].replace("Z", "+00:00"))
    return created_at.astimezone(UTC).date(), int(metadata["stargazers_count"])


def _parse_history(payload: Any, repository: str) -> dict[date, int]:
    if not isinstance(payload, dict) or payload.get("repository") != repository:
        raise ValueError("star history belongs to a different repository")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("star history observations must be a list")

    parsed: dict[date, int] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("invalid star history observation")
        observed_on = date.fromisoformat(str(observation["date"]))
        star_count = int(observation["stars"])
        if star_count < 0:
            raise ValueError("star counts cannot be negative")
        parsed[observed_on] = star_count
    return parsed


def _load_history(
    repository: str,
    seed_path: Path | None,
    published_url: str | None,
) -> dict[date, int]:
    history: dict[date, int] = {}
    if seed_path is not None:
        history.update(
            _parse_history(
                json.loads(seed_path.read_text(encoding="utf-8")), repository
            )
        )
    if published_url:
        try:
            published = _read_json_url(published_url)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            print("Published star history is unavailable; continuing from the seed")
        else:
            history.update(_parse_history(published, repository))
    return history


def _x_coordinate(
    value: date, start: date, span_days: int, left: int, width: int
) -> float:
    return left + ((value - start).days / span_days) * width


def _y_coordinate(value: int, maximum: int, top: int, height: int) -> float:
    return top + height - (value / maximum) * height


def _date_label(value: date) -> str:
    return f"{value:%b} {value.day}, {value.year}"


def render_star_history_svg(
    repository: str,
    observations: list[tuple[date, int]],
    generated_at: date,
) -> str:
    width = 960
    height = 360
    chart_left = 78
    chart_top = 112
    chart_width = 830
    chart_height = 180
    chart_bottom = chart_top + chart_height

    start = observations[0][0]
    end = observations[-1][0]
    if end <= start:
        end = start + timedelta(days=1)
    span_days = max(1, (end - start).days)
    final_count = observations[-1][1]
    y_maximum = max(1, *(value for _, value in observations))

    first_day, first_count = observations[0]
    path_parts = [
        "M",
        f"{_x_coordinate(first_day, start, span_days, chart_left, chart_width):.2f}",
        f"{_y_coordinate(first_count, y_maximum, chart_top, chart_height):.2f}",
    ]
    previous_count = first_count
    for observed_on, count in observations[1:]:
        x = _x_coordinate(observed_on, start, span_days, chart_left, chart_width)
        path_parts.extend(
            [
                "L",
                f"{x:.2f}",
                f"{_y_coordinate(previous_count, y_maximum, chart_top, chart_height):.2f}",
                "L",
                f"{x:.2f}",
                f"{_y_coordinate(count, y_maximum, chart_top, chart_height):.2f}",
            ]
        )
        previous_count = count

    y_ticks = sorted({0, y_maximum // 2, y_maximum})
    grid_lines = []
    for tick in y_ticks:
        y = _y_coordinate(tick, y_maximum, chart_top, chart_height)
        grid_lines.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_left + chart_width}" '
            f'y2="{y:.2f}" stroke="#dbe4ef" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{chart_left - 16}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="13" fill="#64748b">{tick}</text>'
        )

    midpoint = start + timedelta(days=span_days // 2)
    x_labels = []
    for value, anchor in ((start, "start"), (midpoint, "middle"), (end, "end")):
        x = _x_coordinate(value, start, span_days, chart_left, chart_width)
        x_labels.append(
            f'<text x="{x:.2f}" y="{chart_bottom + 32}" text-anchor="{anchor}" '
            f'font-size="13" fill="#64748b">{escape(_date_label(value))}</text>'
        )

    final_x = _x_coordinate(
        observations[-1][0], start, span_days, chart_left, chart_width
    )
    final_y = _y_coordinate(final_count, y_maximum, chart_top, chart_height)
    repository_label = escape(repository)
    updated_label = escape(_date_label(generated_at))
    star_label = "star" if final_count == 1 else "stars"
    path = " ".join(path_parts)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">VoiceCut GitHub stars over time</title>
  <desc id="description">{repository_label} has {final_count} GitHub {star_label} as of {updated_label}.</desc>
  <rect width="{width}" height="{height}" rx="18" fill="#ffffff"/>
  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="17.5" fill="none" stroke="#dbe4ef"/>
  <text x="48" y="50" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="#172033">GitHub stars over time</text>
  <text x="48" y="78" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" fill="#64748b">{repository_label} · updated {updated_label} · {final_count} {star_label}</text>
  <g font-family="Inter, ui-sans-serif, system-ui, sans-serif">
    {"".join(grid_lines)}
    {"".join(x_labels)}
    <path d="{path}" fill="none" stroke="#087b65" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="{final_x:.2f}" cy="{final_y:.2f}" r="6" fill="#087b65" stroke="#ffffff" stroke-width="3"/>
  </g>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the repository star-history chart for GitHub Pages"
    )
    parser.add_argument("--repository", required=True, help="GitHub owner/repository")
    parser.add_argument("--output", required=True, type=Path, help="output SVG path")
    parser.add_argument(
        "--history-output", required=True, type=Path, help="updated history JSON path"
    )
    parser.add_argument("--history-url", help="previously published history JSON URL")
    parser.add_argument("--seed", type=Path, help="repository history seed JSON")
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="environment variable containing the GitHub token",
    )
    args = parser.parse_args()

    token = os.environ.get(args.token_env) or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError(f"{args.token_env} is required to read repository metadata")

    created_at, current_count = _repository_snapshot(args.repository, token)
    history = _load_history(args.repository, args.seed, args.history_url)
    history.setdefault(created_at, 0)
    generated_at = datetime.now(UTC).date()
    history[generated_at] = current_count
    observations = sorted(history.items())

    svg = render_star_history_svg(args.repository, observations, generated_at)
    history_payload = {
        "repository": args.repository,
        "observations": [
            {"date": observed_on.isoformat(), "stars": count}
            for observed_on, count in observations
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    star_label = "star" if current_count == 1 else "stars"
    print(f"Star history built at {args.output} ({current_count} {star_label})")


if __name__ == "__main__":
    main()
