from datetime import datetime, timezone

def to_time_left(deadline: str | datetime) -> str | None:
    """
    Calculate time left until deadline.

    Returns: formatted string if deadline is in the future, otherwise None.
    """
    return _to_time_left(deadline, datetime.now(timezone.utc))


def _to_time_left(deadline: str | datetime, now: datetime) -> str | None:
    if isinstance(deadline, str):
        try:
            d = datetime.fromisoformat(deadline)
        except ValueError:
            return None
    else:
        d = deadline

    if d <= now:
        return 'ended'
        
    delta = d - now
    days = delta.days
    hours = delta.seconds // 3600
    return f"{days} {'day' if days == 1 else 'days'} {hours} {'hour' if hours == 1 else 'hours'} remaining"


def format_datetime(dt: datetime | str) -> str:
    """
    Common formatting for datetime objects.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    return dt.strftime('%Y-%m-%d %H:%M UTC')
