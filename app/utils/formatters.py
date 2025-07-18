import time

from datetime import datetime, timezone


def get_timestamp(with_nanoseconds=False) -> str:
    dt = datetime.now(timezone.utc)
    if with_nanoseconds:
        nanoseconds = time.time_ns() % 1_000_000_000  # Extract nanoseconds
        return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{nanoseconds:09d}"
    return dt.strftime("%Y-%m-%d %H:%M:%S")
