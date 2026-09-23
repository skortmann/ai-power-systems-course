"""The course-wide leaderboard.

Tutorials 01-09 forecast day-ahead load on the same series and the same test
PERIOD. Each one appends its result here, so tutorial 10 can print a single
table showing persistence -> linear -> gradient boosting -> MLP -> LSTM ->
Transformer -> zero-shot foundation model without anybody re-running six
notebooks.

**They are not all evaluated on the same test SAMPLE, and the table says so.**
The feature-based tutorials (01, 02) score on ``make_supervised`` forecast
origins; the sequence tutorials (03, 06) score on ``make_windows`` windows,
which start later because each needs 168 hours of leading context. Recovering
each row's reference from ``RMSE / (1 - Skill)`` shows two distinct baselines:
about 2,952 for 01/02 and about 3,018 for 03/06.

That makes MAE and RMSE comparable to within the difference between two
overlapping samples of the same period, and it makes **Skill not comparable
across those groups at all**, because the denominator differs. The
``Reference_RMSE`` column exists so a reader can see which baseline a row was
scored against instead of having to derive it.

Results are stored as JSON under ``data/artifacts/``. Re-running a notebook
overwrites that notebook's own entries and leaves the others alone, so the
table survives an out-of-order run — but a student who has not run tutorial 03
simply sees no LSTM row, rather than a fabricated one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ARTIFACT_DIR

__all__ = ["ResultEntry", "Leaderboard", "LEADERBOARD_PATH", "record", "leaderboard_table"]

LEADERBOARD_PATH = ARTIFACT_DIR / "leaderboard.json"


@dataclass
class ResultEntry:
    """One model's measured performance on the shared benchmark."""

    model: str
    tutorial: int
    task: str
    metrics: dict[str, float]
    n_parameters: int | None = None
    train_seconds: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def key(self) -> str:
        return f"{self.task}::{self.tutorial:02d}::{self.model}"


class Leaderboard:
    """A tiny JSON-backed store. Deliberately not a database."""

    def __init__(self, path: Path = LEADERBOARD_PATH) -> None:
        self.path = path
        self.entries: dict[str, ResultEntry] = {}
        self.load()

    def load(self) -> Leaderboard:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.entries = {k: ResultEntry(**v) for k, v in raw.items()}
        return self

    def save(self) -> Leaderboard:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in sorted(self.entries.items())}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self

    def add(self, entry: ResultEntry) -> Leaderboard:
        self.entries[entry.key] = entry
        return self.save()

    def to_frame(self, task: str | None = None) -> pd.DataFrame:
        rows = [e for e in self.entries.values() if task is None or e.task == task]
        if not rows:
            return pd.DataFrame(columns=["Tutorial", "Model"])
        records = []
        for entry in sorted(rows, key=lambda e: (e.tutorial, e.model)):
            record_ = {"Tutorial": entry.tutorial, "Model": entry.model, **entry.metrics}
            if entry.n_parameters is not None:
                record_["Params"] = entry.n_parameters
            if entry.train_seconds is not None:
                record_["Train_s"] = round(entry.train_seconds, 1)
            if entry.notes:
                record_["Notes"] = entry.notes
            records.append(record_)
        return pd.DataFrame(records).set_index(["Tutorial", "Model"])

    def clear(self, tutorial: int | None = None) -> Leaderboard:
        """Drop all entries, or only those of one tutorial."""
        if tutorial is None:
            self.entries = {}
        else:
            self.entries = {k: v for k, v in self.entries.items() if v.tutorial != tutorial}
        return self.save()


def record(
    model: str,
    tutorial: int,
    metrics: dict[str, float],
    task: str = "load_day_ahead",
    **kwargs: Any,
) -> ResultEntry:
    """Convenience wrapper: build an entry, persist it, return it.

    ``record("LSTM", 3, point_metrics(y_test, y_hat), n_parameters=12_345)``
    """
    entry = ResultEntry(model=model, tutorial=tutorial, task=task, metrics=metrics, **kwargs)
    Leaderboard().add(entry)
    return entry


def leaderboard_table(task: str = "load_day_ahead", sort_by: str = "MAE") -> pd.DataFrame:
    """Return the current leaderboard, best first where the metric allows it.

    Adds ``Reference_RMSE``, recovered as ``RMSE / (1 - Skill)``. Rows whose
    reference differs were scored against a different baseline, so their Skill
    values are not comparable with one another -- see the module docstring.
    Sorting defaults to MAE rather than Skill for exactly that reason.
    """
    frame = Leaderboard().to_frame(task=task)
    if {"RMSE", "Skill"} <= set(frame.columns):
        skill = frame["Skill"]
        frame["Reference_RMSE"] = (frame["RMSE"] / (1.0 - skill)).where(
            skill.notna() & (skill != 1.0)
        )
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by)
    return frame.round(3)


def reference_groups(task: str = "load_day_ahead", tolerance: float = 1.0) -> dict:
    """Group leaderboard rows by the baseline their Skill was measured against.

    More than one group means Skill cannot be read down the column as a
    ranking. Returns ``{rounded reference RMSE: [model names]}``.
    """
    frame = leaderboard_table(task=task)
    if "Reference_RMSE" not in frame.columns:
        return {}
    groups: dict[float, list[str]] = {}
    for (_tutorial, model), reference in frame["Reference_RMSE"].items():
        if pd.isna(reference):
            continue
        bucket = next(
            (key for key in groups if abs(key - reference) <= tolerance), round(reference, 1)
        )
        groups.setdefault(bucket, []).append(model)
    return groups
