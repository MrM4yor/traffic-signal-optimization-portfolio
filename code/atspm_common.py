"""Shared, dependency-free helpers for ATSPM reconstruction."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_timestamp(value: str) -> datetime:
    # .NET exports can contain seven fractional digits; Python accepts six.
    """Parse exported local-clock timestamps, truncating fractional precision to Python microseconds. No timezone conversion is performed."""
    head, dot, fraction = value.strip().partition(".")
    if not dot:
        return datetime.strptime(head, TIMESTAMP_FORMAT)
    return datetime.strptime(
        f"{head}.{fraction[:6].ljust(6, '0')}", f"{TIMESTAMP_FORMAT}.%f"
    )


def seconds_since(day_start: datetime, value: str) -> float:
    """Return elapsed seconds against the caller's day-start clock; both timestamps must use the same time basis."""
    return (parse_timestamp(value) - day_start).total_seconds()


def read_events(path: Path) -> Iterable[tuple[int, int, datetime]]:
    """Stream validated CSV rows as event code, channel/parameter, and local-clock timestamp."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"eventCode", "eventParam", "timestamp"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            yield int(row["eventCode"]), int(row["eventParam"]), parse_timestamp(
                row["timestamp"]
            )


class TableParser(HTMLParser):
    """Extract HTML tables without adding BeautifulSoup/pandas as dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def parse_tables(path: Path) -> list[list[list[str]]]:
    """Extract tables from cached HTML; this helper never requests the network."""
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.tables


def table_as_dicts(table: list[list[str]]) -> list[dict[str, str]]:
    """Map data rows to the first header row, ignoring rows with inconsistent column counts."""
    header = table[0]
    return [dict(zip(header, row)) for row in table[1:] if len(row) == len(header)]


def dump_json(path: Path, value) -> None:
    """Write UTF-8 JSON after creating parent directories; the caller chooses a new output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def event_profile(path: Path) -> dict:
    """Summarize event codes and channel parameters without assuming unknown vendor events are vehicle counts."""
    codes: Counter[int] = Counter()
    params: dict[int, set[int]] = defaultdict(set)
    first = last = None
    for code, param, timestamp in read_events(path):
        codes[code] += 1
        params[code].add(param)
        first = timestamp if first is None else min(first, timestamp)
        last = timestamp if last is None else max(last, timestamp)
    return {
        "rows": sum(codes.values()),
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
        "codes": {
            str(code): {"count": count, "parameters": sorted(params[code])}
            for code, count in sorted(codes.items())
        },
    }
