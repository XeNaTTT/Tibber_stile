"""Dependency-free data helpers for the energy chart."""

import datetime as dt
import logging
import statistics

PRICE_INTERVAL = dt.timedelta(minutes=15)
QUARTER_SLOTS_PER_DAY = 96


def quarter_slot_index(timestamp):
    """Return the quarter-hour slot (0..95) containing ``timestamp``."""
    return timestamp.hour * 4 + timestamp.minute // 15


def x_for_quarter_slot(panel_x, panel_width, slot):
    """Map a day-local quarter slot to the shared chart coordinate system.

    ``slot`` may be 96 when drawing the right edge of the last price interval.
    Using 96 intervals, rather than spacing 96 samples over 95 intervals, keeps
    the day boundary at exactly ``panel_x + panel_width``.
    """
    return panel_x + max(0, min(QUARTER_SLOTS_PER_DAY, slot)) * panel_width / QUARTER_SLOTS_PER_DAY


def consumption_to_quarter_series(nodes, day, local_tz, resolution):
    """Place Tibber consumption readings on a sparse, 96-slot daily grid.

    Tibber reports energy per requested interval in kWh.  It is converted to
    average watts for that interval.  Hourly readings deliberately occupy only
    their real hour boundary; missing quarter-hours remain ``None`` and may
    only be bridged by the renderer as a visual line interpolation.
    """
    values = [None] * QUARTER_SLOTS_PER_DAY
    timestamps = [None] * QUARTER_SLOTS_PER_DAY
    interval_hours = 0.25 if resolution == "15min" else 1.0
    for node in nodes or []:
        try:
            timestamp = dt.datetime.fromisoformat(node["from"]).astimezone(local_tz)
            value = float(node["consumption"])
        except (KeyError, TypeError, ValueError):
            continue
        if timestamp.date() != day or not (value >= 0):
            continue
        slot = quarter_slot_index(timestamp)
        values[slot] = value * 1000.0 / interval_hours
        timestamps[slot] = timestamp
    return timestamps, values


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


def merge_consumption_series(historical, local):
    """Overlay genuine local intervals without filling either sparse source."""
    size = max(len(historical or []), len(local or []), QUARTER_SLOTS_PER_DAY)
    merged = [None] * size
    for index in range(size):
        old = historical[index] if historical is not None and index < len(historical) else None
        pulse = local[index] if local is not None and index < len(local) else None
        merged[index] = pulse if pulse is not None else old
    return merged
