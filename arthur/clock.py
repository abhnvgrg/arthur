from __future__ import annotations

import os
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ClockError(ValueError):
    pass


def local_zone() -> tzinfo:
    name = os.getenv("ARTHUR_TIMEZONE")
    if not name:
        return datetime.now().astimezone().tzinfo
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ClockError(f"ARTHUR_TIMEZONE is not a known timezone: {name!r}")


def now_local() -> datetime:
    return datetime.now(local_zone())


def zone_name() -> str:
    return os.getenv("ARTHUR_TIMEZONE") or str(now_local().tzname())


def describe_now() -> str:
    moment = now_local()
    return f"{moment.strftime('%A %d %B %Y, %H:%M')} ({zone_name()})"
