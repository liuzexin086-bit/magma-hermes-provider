"""
Temporal Parser — Resolves relative time expressions into absolute timestamps.

Handles patterns like "last Friday", "yesterday", "2 hours ago", "May 2026".
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Map English day names → weekday index (Monday=0)
_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Regex patterns in priority order
_PATTERNS = [
    # "X days/weeks/months ago"
    (r"(?P<num>\d+)\s+(day|days|week|weeks|month|months)\s+ago", "ago"),
    # "last N days/weeks/months"
    (r"last\s+(?P<num>\d+)\s+(day|days|week|weeks|month|months)", "last_n"),
    # "yesterday"
    (r"\byesterday\b", "yesterday"),
    # "today"
    (r"\btoday\b", "today"),
    # "last weekday" e.g. "last friday"
    (r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "last_weekday"),
    # "this weekday" e.g. "this monday"
    (r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "this_weekday"),
    # "Month YYYY" e.g. "May 2026"
    (r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", "month_year"),
    # ISO date 2026-05-15
    (r"(\d{4})-(\d{2})-(\d{2})", "iso_date"),
    # MM/DD/YYYY
    (r"(\d{1,2})/(\d{1,2})/(\d{4})", "slash_date"),
]


class TemporalParser:
    """Parse relative and absolute time references in queries."""

    def __init__(self, now: Optional[datetime] = None):
        self._now = now or datetime.now()

    def parse(self, text: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Return (start, end) time window if a temporal expression is found.
        Returns None if no temporal expression is detected.
        """
        text_lower = text.lower().strip()
        for pattern, kind in _PATTERNS:
            m = re.search(pattern, text_lower)
            if not m:
                continue
            result = self._resolve(kind, m)
            if result:
                logger.debug("TemporalParser: '%s' -> %s", m.group(0), result)
                return result
        return None

    def _resolve(self, kind: str, m: re.Match) -> Optional[Tuple[datetime, datetime]]:
        now = self._now
        if kind == "yesterday":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end = start + timedelta(days=1)
            return (start, end)

        if kind == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return (start, end)

        if kind == "ago":
            num = int(m.group("num"))
            unit = m.group(1).lower()
            delta = {
                "day": timedelta(days=num),
                "days": timedelta(days=num),
                "week": timedelta(weeks=num),
                "weeks": timedelta(weeks=num),
                "month": timedelta(days=num * 30),
                "months": timedelta(days=num * 30),
            }.get(unit, timedelta(days=num))
            end = now
            start = now - delta
            return (start, end)

        if kind == "last_n":
            num = int(m.group("num"))
            unit = m.group(1).lower()
            delta = {
                "day": timedelta(days=num),
                "days": timedelta(days=num),
                "week": timedelta(weeks=num),
                "weeks": timedelta(weeks=num),
                "month": timedelta(days=num * 30),
                "months": timedelta(days=num * 30),
            }.get(unit, timedelta(days=num))
            end = now
            start = now - delta
            return (start, end)

        if kind == "last_weekday":
            day_name = m.group(1).lower()
            target = _DAY_NAMES.get(day_name)
            if target is None:
                return None
            current_weekday = now.weekday()
            days_back = (current_weekday - target) % 7
            if days_back == 0:
                days_back = 7  # last week not today
            start = (now - timedelta(days=days_back)).replace(
                hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return (start, end)

        if kind == "this_weekday":
            day_name = m.group(1).lower()
            target = _DAY_NAMES.get(day_name)
            if target is None:
                return None
            current_weekday = now.weekday()
            if current_weekday <= target:
                days_forward = target - current_weekday
            else:
                days_forward = 7 - (current_weekday - target)
            start = (now + timedelta(days=days_forward)).replace(
                hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return (start, end)

        if kind == "month_year":
            month_name = m.group(1).lower()
            year = int(m.group(2))
            month = _MONTH_NAMES.get(month_name)
            if month is None:
                return None
            # approximate
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
            return (start, end)

        if kind == "iso_date":
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start = datetime(y, mo, d)
            end = start + timedelta(days=1)
            return (start, end)

        if kind == "slash_date":
            mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start = datetime(y, mo, d)
            end = start + timedelta(days=1)
            return (start, end)

        return None
