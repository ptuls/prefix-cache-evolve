"""Generate the technical report's incumbent evaluation-trajectory figure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click


@dataclass(frozen=True)
class JourneyRun:
    """One retained discovery-verifier search run."""

    label: str
    snapshot_path: str


JOURNEY_RUNS = (
    JourneyRun("R1", "runs/20260607_151718_996826/snapshot.json"),
    JourneyRun("R2", "runs/20260607_205745_434378/snapshot.json"),
    JourneyRun("R3", "runs/20260607_212037_133653/snapshot.json"),
    JourneyRun("R4", "runs/20260607_215647_118596/snapshot.json"),
    JourneyRun("R5", "runs/20260607_231959_613687/snapshot.json"),
    JourneyRun("R6", "runs/20260608_093834_121786/snapshot.json"),
)
PARADIGM_EVALUATIONS = (30, 90, 130, 160, 190, 240)


def _load_snapshot(path: Path) -> dict[str, Any]:
    """Load one Levi snapshot."""
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_improvements(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return strict best-score improvements in evaluation order."""
    improvements = []
    best = float("-inf")
    for entry in history:
        score = float(entry["best_score"])
        if score > best:
            improvements.append(entry)
            best = score
    return improvements


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate the score-history accounting used by the figure."""
    run_state = snapshot["run_state"]
    history = snapshot["score_history"]
    eval_count = int(run_state["eval_count"])
    error_count = int(run_state["error_count"])
    accept_count = int(run_state["accept_count"])
    scored_accept_count = sum(bool(entry["accepted"]) for entry in history)
    if len(history) + error_count != eval_count:
        raise ValueError("score history and error count do not cover all evaluations")
    if scored_accept_count != accept_count:
        raise ValueError("score history acceptance count does not match run state")


def _map(value: float, low: float, high: float, start: float, span: float) -> float:
    """Map one value from a data interval to a plot interval."""
    return start + (value - low) / (high - low) * span


def _point(x: float, y: float) -> str:
    """Format one TikZ coordinate."""
    return f"({x:.3f},{y:.3f})"


def _draw_step(
    points: list[tuple[float, float]],
    *,
    end_x: float,
    color: str = "navy",
) -> str:
    """Draw a best-score-so-far step line."""
    path = [f"\\draw[very thick, draw={color}] {_point(*points[0])}"]
    for previous, current in zip(points, points[1:], strict=False):
        path.append(f" -- {_point(current[0], previous[1])} -- {_point(*current)}")
    path.append(f" -- {_point(end_x, points[-1][1])};")
    return "".join(path)


def _journey_panel(
    snapshots: list[dict[str, Any]],
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> list[str]:
    """Render the cross-run global-best trajectory."""
    total_evaluations = sum(int(snapshot["run_state"]["eval_count"]) for snapshot in snapshots)
    y_low = 74.2
    y_high = 77.35
    lines = [
        f"\\node[anchor=west, font=\\bfseries\\scriptsize] at "
        f"{_point(left, bottom + height + 0.35)} "
        "{A. Retained discovery-search journey};",
    ]

    offset = 0
    for index, (run, snapshot) in enumerate(zip(JOURNEY_RUNS, snapshots, strict=True)):
        run_evaluations = int(snapshot["run_state"]["eval_count"])
        x0 = _map(offset, 0, total_evaluations, left, width)
        x1 = _map(offset + run_evaluations, 0, total_evaluations, left, width)
        fill = "lightblue" if index % 2 == 0 else "graybg"
        lines.append(
            f"\\fill[{fill}, opacity=0.55] {_point(x0, bottom)} rectangle "
            f"{_point(x1, bottom + height)};"
        )
        lines.append(
            f"\\draw[darkgray, densely dashed] {_point(x1, bottom)} -- "
            f"{_point(x1, bottom + height)};"
        )
        lines.append(
            f"\\node[font=\\scriptsize, anchor=north, text=darkgray] at "
            f"{_point((x0 + x1) / 2, bottom + height - 0.05)} {{{run.label}}};"
        )
        offset += run_evaluations

    for score in (74.5, 75.0, 75.5, 76.0, 76.5, 77.0):
        y = _map(score, y_low, y_high, bottom, height)
        lines.extend(
            [
                f"\\draw[darkgray!20] {_point(left, y)} -- {_point(left + width, y)};",
                f"\\node[font=\\scriptsize, anchor=east] at {_point(left - 0.08, y)} "
                f"{{{score:.1f}}};",
            ]
        )

    global_improvements: list[tuple[int, float, str]] = []
    offset = 0
    best = float("-inf")
    for run, snapshot in zip(JOURNEY_RUNS, snapshots, strict=True):
        history = list(snapshot["score_history"])
        for entry in _strict_improvements(history):
            score = float(entry["best_score"])
            if score > best:
                global_improvements.append((offset + int(entry["eval_number"]), score, run.label))
                best = score
        offset += int(snapshot["run_state"]["eval_count"])

    step_points = [
        (
            _map(evaluation, 0, total_evaluations, left, width),
            _map(score, y_low, y_high, bottom, height),
        )
        for evaluation, score, _ in global_improvements
    ]
    lines.append(_draw_step(step_points, end_x=left + width))
    for evaluation, score, _ in global_improvements:
        x = _map(evaluation, 0, total_evaluations, left, width)
        y = _map(score, y_low, y_high, bottom, height)
        lines.append(f"\\fill[orange] {_point(x, y)} circle (0.055);")

    annotations = (
        (1, 74.32139579831396, "compact seed", 0.25, 0.42, "west"),
        (403, 76.06931118408542, "pressure-aware +1.399", -0.1, 0.4, "south"),
        (811, 76.62991069358938, "composed seed +0.509", -0.2, 0.42, "south east"),
        (833, 77.08037703140361, "scan penalty +0.450", -0.35, -0.38, "north east"),
        (891, 77.22963163298509, "promoted +0.149", 0.18, -0.08, "north west"),
    )
    for evaluation, score, label, dx, dy, anchor in annotations:
        x = _map(evaluation, 0, total_evaluations, left, width)
        y = _map(score, y_low, y_high, bottom, height)
        lines.append(
            f"\\node[font=\\scriptsize, anchor={anchor}, fill=white, inner sep=1.2pt] at "
            f"{_point(x + dx, y + dy)} {{{label}}};"
        )

    for evaluation in (0, 214, 512, 810, total_evaluations):
        x = _map(evaluation, 0, total_evaluations, left, width)
        lines.extend(
            [
                f"\\draw[darkgray] {_point(x, bottom)} -- {_point(x, bottom - 0.08)};",
                f"\\node[font=\\scriptsize, anchor=north] at {_point(x, bottom - 0.11)} "
                f"{{{evaluation}}};",
            ]
        )
    lines.extend(
        [
            f"\\draw[darkgray, thick] {_point(left, bottom)} -- {_point(left + width, bottom)};",
            f"\\node[font=\\scriptsize, anchor=north] at "
            f"{_point(left + width / 2, bottom - 0.42)} {{cumulative evaluation index}};",
        ]
    )
    return lines


def _final_run_panel(
    snapshot: dict[str, Any],
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> list[str]:
    """Render the detailed trajectory and evaluation-outcome raster."""
    eval_count = int(snapshot["run_state"]["eval_count"])
    history = list(snapshot["score_history"])
    scored_evaluations = {int(entry["eval_number"]) for entry in history}
    accepted_evaluations = {
        int(entry["eval_number"]) for entry in history if bool(entry["accepted"])
    }
    error_evaluations = set(range(1, eval_count + 1)) - scored_evaluations
    improvements = _strict_improvements(history)
    y_low = 72.0
    y_high = 77.4
    lines = [
        f"\\node[anchor=west, font=\\bfseries\\scriptsize] at "
        f"{_point(left, bottom + height + 0.35)} "
        "{B. Final targeted run: scored candidates and verification outcomes};",
    ]

    flat_x = _map(81, 0, eval_count, left, width)
    lines.extend(
        [
            f"\\fill[graybg] {_point(flat_x, bottom)} rectangle "
            f"{_point(left + width, bottom + height)};",
            f"\\node[font=\\scriptsize, anchor=north west, text=darkgray] at "
            f"{_point(flat_x + 0.08, bottom + height - 0.52)} "
            "{217-evaluation flat tail};",
        ]
    )

    for score in (72, 73, 74, 75, 76, 77):
        y = _map(score, y_low, y_high, bottom, height)
        lines.extend(
            [
                f"\\draw[darkgray!20] {_point(left, y)} -- {_point(left + width, y)};",
                f"\\node[font=\\scriptsize, anchor=east] at {_point(left - 0.08, y)} {{{score}}};",
            ]
        )

    for evaluation in PARADIGM_EVALUATIONS:
        x = _map(evaluation, 0, eval_count, left, width)
        lines.append(
            f"\\draw[orange, densely dashed] {_point(x, bottom)} -- {_point(x, bottom + height)};"
        )
    lines.append(
        f"\\node[font=\\scriptsize, anchor=north east, text=orange] at "
        f"{_point(left + width - 0.05, bottom + height - 0.48)} "
        "{GPT-5.5 paradigm triggers; all static-rejected};"
    )

    for entry in history:
        evaluation = int(entry["eval_number"])
        score = float(entry["score"])
        x = _map(evaluation, 0, eval_count, left, width)
        if score < y_low:
            y = bottom + 0.04
            lines.append(
                f"\\fill[red] {_point(x, y)} -- {_point(x - 0.045, y + 0.11)} -- "
                f"{_point(x + 0.045, y + 0.11)} -- cycle;"
            )
            continue
        y = _map(score, y_low, y_high, bottom, height)
        color = "green" if bool(entry["accepted"]) else "darkgray!55"
        radius = "0.045" if bool(entry["accepted"]) else "0.032"
        lines.append(f"\\fill[{color}] {_point(x, y)} circle ({radius});")

    step_points = [
        (
            _map(int(entry["eval_number"]), 0, eval_count, left, width),
            _map(float(entry["best_score"]), y_low, y_high, bottom, height),
        )
        for entry in improvements
    ]
    lines.append(_draw_step(step_points, end_x=left + width))
    for entry in improvements:
        x = _map(int(entry["eval_number"]), 0, eval_count, left, width)
        y = _map(float(entry["best_score"]), y_low, y_high, bottom, height)
        lines.append(f"\\fill[orange] {_point(x, y)} circle (0.06);")

    annotations = (
        (1, 76.62991069358938, "eval 1: seed 76.630", 0.18, -0.34, "north west"),
        (23, 77.08037703140361, "eval 23: +0.450", 0.18, -0.18, "north west"),
        (81, 77.22963163298509, "eval 81: +0.149 incumbent", 0.15, -0.18, "north west"),
    )
    for evaluation, score, label, dx, dy, anchor in annotations:
        x = _map(evaluation, 0, eval_count, left, width)
        y = _map(score, y_low, y_high, bottom, height)
        lines.append(
            f"\\node[font=\\scriptsize, anchor={anchor}, fill=white, inner sep=1.2pt] at "
            f"{_point(x + dx, y + dy)} {{{label}}};"
        )

    lines.append(
        f"\\node[font=\\scriptsize, anchor=south west, text=red] at "
        f"{_point(left + 0.08, bottom + 0.05)} {{8 scored candidates below plot range}};"
    )

    outcome_rows = (
        ("scored 61", scored_evaluations, "blue", bottom - 0.55),
        ("accepted 18", accepted_evaluations, "green", bottom - 0.90),
        ("errors 237", error_evaluations, "red", bottom - 1.25),
    )
    for label, evaluations, color, y in outcome_rows:
        lines.append(
            f"\\node[font=\\scriptsize, anchor=east] at {_point(left - 0.08, y)} {{{label}}};"
        )
        lines.append(f"\\draw[darkgray!20] {_point(left, y)} -- {_point(left + width, y)};")
        for evaluation in sorted(evaluations):
            x = _map(evaluation, 0, eval_count, left, width)
            lines.append(
                f"\\draw[{color}, line width=0.35pt] {_point(x, y - 0.08)} -- "
                f"{_point(x, y + 0.08)};"
            )

    for evaluation in (0, 50, 100, 150, 200, 250, eval_count):
        x = _map(evaluation, 0, eval_count, left, width)
        lines.extend(
            [
                f"\\draw[darkgray] {_point(x, bottom)} -- {_point(x, bottom - 0.08)};",
                f"\\node[font=\\scriptsize, anchor=north] at {_point(x, bottom - 0.11)} "
                f"{{{evaluation}}};",
            ]
        )
    lines.extend(
        [
            f"\\draw[darkgray, thick] {_point(left, bottom)} -- {_point(left + width, bottom)};",
            f"\\node[font=\\scriptsize, anchor=north] at "
            f"{_point(left + width / 2, bottom - 1.52)} {{evaluation index}};",
        ]
    )
    return lines


def render_trajectory(repo_root: Path) -> str:
    """Render the retained incumbent trajectory as a TikZ fragment."""
    snapshots = [_load_snapshot(repo_root / run.snapshot_path) for run in JOURNEY_RUNS]
    for snapshot in snapshots:
        _validate_snapshot(snapshot)
    lines = [
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        *_journey_panel(snapshots, left=1.55, bottom=6.2, width=14.4, height=3.15),
        *_final_run_panel(snapshots[-1], left=1.55, bottom=1.9, width=14.4, height=3.0),
        "\\end{tikzpicture}%",
        "}",
        "",
    ]
    return "\n".join(lines)


@click.command()
@click.option(
    "--repo-root",
    type=click.Path(path_type=Path),
    default=Path.cwd,
    show_default="current directory",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("docs/figures/incumbent_eval_trajectory.tex"),
    show_default=True,
)
def main(repo_root: Path, output: Path) -> None:
    """Generate the report figure."""
    output = output if output.is_absolute() else repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trajectory(repo_root), encoding="utf-8")
    click.echo(output)


if __name__ == "__main__":
    main()
