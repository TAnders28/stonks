"""Minimal NYSE session calendar (weekends + full-day holidays 2024-2027). No network.
Used to decide the expected last *completed* daily bar, so partial intraday bars never enter data/daily."""
from __future__ import annotations
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
NYSE_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025 (incl. Jan 9 national day of mourning)
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
# Daily bar treated as final this long after the 16:00 ET close (Yahoo settles ~minutes later)
FINAL_AFTER = time(16, 30)


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS


def prev_session(d: date) -> date:
    d = d - timedelta(days=1)
    while not is_session(d):
        d -= timedelta(days=1)
    return d


def next_session(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_session(d):
        d += timedelta(days=1)
    return d


def expected_last_complete_session(now: datetime | None = None) -> date:
    now = (now or datetime.now(ET)).astimezone(ET)
    d = now.date()
    if is_session(d) and now.time() >= FINAL_AFTER:
        return d
    return prev_session(d)
