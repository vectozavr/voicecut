from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

GITHUB_API_VERSION = "2026-03-10"


def _github_json(url: str, token: str) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.star+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "voicecut-star-history",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return payload, dict(response.headers.items())


def _load_repository_history(repository: str, token: str) -> tuple[date, list[date]]:
    encoded_repository = urllib.parse.quote(repository, safe="/")
    metadata, _ = _github_json(
        f"https://api.github.com/repos/{encoded_repository}", token
    )
    created_at = datetime.fromisoformat(metadata["created_at"].replace("Z", "+00:00"))
    expected_star_count = int(metadata["stargazers_count"])

    star_dates: list[date] = []
    page = 1
    while True:
        stargazers, _ = _github_json(
            "https://api.github.com/repos/"
            f"{encoded_repository}/stargazers?per_page=100&page={page}",
            token,
        )
        if not isinstance(stargazers, list):
            raise RuntimeError("GitHub returned an invalid stargazer response")
        for stargazer in stargazers:
            timestamp = stargazer.get("starred_at")
            if not timestamp:
                raise RuntimeError(
                    "GitHub did not return stargazer timestamps; the workflow token "
                    "must be allowed to read repository metadata"
                )
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            star_dates.append(parsed.astimezone(UTC).date())
        if len(stargazers) < 100:
            break
        page += 1

    if len(star_dates) != expected_star_count:
        raise RuntimeError(
            "GitHub returned incomplete stargazer history: "
            f"expected {expected_star_count}, received {len(star_dates)}"
        )

    return created_at.astimezone(UTC).date(), sorted(star_dates)


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
    created_at: date,
    star_dates: list[date],
    generated_at: date,
) -> str:
    width = 960
    height = 360
    chart_left = 78
    chart_top = 112
    chart_width = 830
    chart_height = 180
    chart_bottom = chart_top + chart_height

    start = min([created_at, *star_dates, generated_at])
    end = max([created_at, *star_dates, generated_at])
    if end <= start:
        end = start + timedelta(days=1)
    span_days = max(1, (end - start).days)
    final_count = len(star_dates)
    y_maximum = max(1, final_count)

    daily_stars = Counter(star_dates)
    path_parts = [
        "M",
        f"{_x_coordinate(start, start, span_days, chart_left, chart_width):.2f}",
        f"{_y_coordinate(0, y_maximum, chart_top, chart_height):.2f}",
    ]
    cumulative = 0
    for day, added in sorted(daily_stars.items()):
        x = _x_coordinate(day, start, span_days, chart_left, chart_width)
        path_parts.extend(
            [
                "L",
                f"{x:.2f}",
                f"{_y_coordinate(cumulative, y_maximum, chart_top, chart_height):.2f}",
            ]
        )
        cumulative += added
        path_parts.extend(
            [
                "L",
                f"{x:.2f}",
                f"{_y_coordinate(cumulative, y_maximum, chart_top, chart_height):.2f}",
            ]
        )
    path_parts.extend(
        [
            "L",
            f"{_x_coordinate(end, start, span_days, chart_left, chart_width):.2f}",
            f"{_y_coordinate(cumulative, y_maximum, chart_top, chart_height):.2f}",
        ]
    )

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

    final_x = _x_coordinate(end, start, span_days, chart_left, chart_width)
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
        "--token-env",
        default="GITHUB_TOKEN",
        help="environment variable containing the GitHub token",
    )
    args = parser.parse_args()

    token = os.environ.get(args.token_env) or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError(
            f"{args.token_env} is required to read timestamped GitHub stargazers"
        )
    created_at, star_dates = _load_repository_history(args.repository, token)
    generated_at = datetime.now(UTC).date()
    svg = render_star_history_svg(
        args.repository,
        created_at,
        star_dates,
        generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    count = len(star_dates)
    star_label = "star" if count == 1 else "stars"
    print(f"Star history built at {args.output} ({count} {star_label})")


if __name__ == "__main__":
    main()
