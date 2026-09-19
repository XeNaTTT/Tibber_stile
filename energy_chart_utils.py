"""Dependency-free data helpers for the energy chart."""

import datetime as dt
import logging
import statistics

PRICE_INTERVAL = dt.timedelta(minutes=15)


def price_slots_to_quarters(slots, local_tz):
    """Return sorted price timestamps and cent values on a strict 15-minute grid."""
    parsed_by_time = {}
    for slot in slots or []:
        try:
            start = dt.datetime.fromisoformat(slot["startsAt"]).astimezone(local_tz)
            parsed_by_time[start] = float(slot["total"]) * 100
        except (KeyError, TypeError, ValueError):
            logging.warning("Ungültiger Tibber-Preisslot übersprungen: %r", slot)
    parsed = sorted(parsed_by_time.items())
    if not parsed:
        return [], []

    deltas = [
        (right[0] - left[0]).total_seconds() / 60
        for left, right in zip(parsed, parsed[1:])
        if 0 < (right[0] - left[0]).total_seconds() <= 2 * 60 * 60
    ]
    nominal_interval = (
        PRICE_INTERVAL
        if deltas and statistics.median(deltas) < 30
        else dt.timedelta(hours=1)
    )

    normalized = []
    for index, (start, price) in enumerate(parsed):
        next_start = parsed[index + 1][0] if index + 1 < len(parsed) else None
        duration = nominal_interval
        if next_start is not None:
            gap = next_start - start
            if PRICE_INTERVAL <= gap <= dt.timedelta(hours=2):
                duration = gap
        cursor = start
        while cursor < start + duration:
            normalized.append((cursor, price))
            cursor += PRICE_INTERVAL

    return [item[0] for item in normalized], [item[1] for item in normalized]


def format_power_peak(watts):
    """Format a consumption peak compactly for the e-paper chart."""
    return f"{watts / 1000:.2f} kW" if watts >= 1000 else f"{watts:.0f} W"
