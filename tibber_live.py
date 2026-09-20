#!/usr/bin/env python3
"""Fetch exactly one Tibber Pulse sample and build wake-up intervals."""

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

GRAPHQL_URL = "https://api.tibber.com/v1-beta/gql"
LOCAL_TZ = ZoneInfo("Europe/Berlin")
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tibber_snapshots.db")
MIN_INTERVAL_SECONDS = 12 * 60
MAX_INTERVAL_SECONDS = 18 * 60
QUARTER_BOUNDARY_TOLERANCE_SECONDS = 3 * 60
MAX_AVERAGE_POWER_W = 100_000
HTTP_TIMEOUT_SECONDS = 15
WEBSOCKET_TIMEOUT_SECONDS = 15


def _load_websocket_connect():
    """Load the current websockets client API, falling back to versions such as 10.4."""
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        try:
            from websockets import connect
        except ImportError as exc:
            raise RuntimeError("Python package 'websockets' is not installed") from exc
    return connect


@dataclass(frozen=True)
class LiveSnapshot:
    timestamp: dt.datetime
    power_w: float | None = None
    average_power_w: float | None = None
    min_power_w: float | None = None
    max_power_w: float | None = None
    accumulated_consumption_kwh: float | None = None
    accumulated_consumption_last_hour_kwh: float | None = None
    last_meter_consumption_kwh: float | None = None
    signal_strength: float | None = None


def _number(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_snapshot(payload):
    """Parse a liveMeasurement object, retaining nullable fields."""
    if not isinstance(payload, dict) or not payload.get("timestamp"):
        raise ValueError("liveMeasurement has no timestamp")
    timestamp = dt.datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("liveMeasurement timestamp is not timezone-aware")
    return LiveSnapshot(
        timestamp=timestamp,
        power_w=_number(payload.get("power")),
        average_power_w=_number(payload.get("averagePower")),
        min_power_w=_number(payload.get("minPower")),
        max_power_w=_number(payload.get("maxPower")),
        accumulated_consumption_kwh=_number(payload.get("accumulatedConsumption")),
        accumulated_consumption_last_hour_kwh=_number(payload.get("accumulatedConsumptionLastHour")),
        last_meter_consumption_kwh=_number(payload.get("lastMeterConsumption")),
        signal_strength=_number(payload.get("signalStrength")),
    )


def select_realtime_home(homes, preferred_home_id=None):
    enabled = [home for home in homes or []
               if ((home or {}).get("features") or {}).get("realTimeConsumptionEnabled") is True]
    if preferred_home_id:
        return next((home for home in enabled if home.get("id") == preferred_home_id), None)
    return enabled[0] if enabled else None


def fetch_live_config(token, preferred_home_id=None, session=None):
    if session is None:
        import requests
        session = requests
    query = """{ viewer { websocketSubscriptionUrl homes {
      id appNickname features { realTimeConsumptionEnabled }
    } } }"""
    response = session.post(
        GRAPHQL_URL, json={"query": query},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(f"Tibber GraphQL error: {body['errors']}")
    viewer = (body.get("data") or {}).get("viewer") or {}
    home = select_realtime_home(viewer.get("homes"), preferred_home_id)
    if not home:
        raise RuntimeError("No Tibber home has real-time consumption enabled")
    websocket_url = viewer.get("websocketSubscriptionUrl")
    if not websocket_url:
        raise RuntimeError("Tibber did not return websocketSubscriptionUrl")
    return websocket_url, home


async def _receive_one_snapshot(websocket_url, home_id, token, timeout):
    connect = _load_websocket_connect()
    query = """subscription LiveMeasurement($homeId: ID!) {
      liveMeasurement(homeId: $homeId) { timestamp power averagePower minPower maxPower
        accumulatedConsumption accumulatedConsumptionLastHour lastMeterConsumption signalStrength }
    }"""
    async with asyncio.timeout(timeout):
        async with connect(websocket_url, subprotocols=["graphql-transport-ws"],
                           open_timeout=timeout, close_timeout=3) as socket:
            await socket.send(json.dumps({"type": "connection_init", "payload": {"token": token}}))
            while True:
                message = json.loads(await socket.recv())
                if message.get("type") == "connection_ack":
                    break
                if message.get("type") in ("connection_error", "error"):
                    raise RuntimeError(f"Tibber WebSocket rejected connection: {message.get('payload')}")
            operation_id = uuid.uuid4().hex
            await socket.send(json.dumps({
                "id": operation_id, "type": "subscribe",
                "payload": {"query": query, "variables": {"homeId": home_id}},
            }))
            while True:
                message = json.loads(await socket.recv())
                if message.get("type") == "ping":
                    await socket.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("type") == "next":
                    payload = ((message.get("payload") or {}).get("data") or {}).get("liveMeasurement")
                    if payload:
                        await socket.send(json.dumps({"id": operation_id, "type": "complete"}))
                        return parse_snapshot(payload)
                if message.get("type") in ("error", "complete"):
                    raise RuntimeError(f"LiveMeasurement ended without data: {message.get('payload')}")


def get_tibber_live_snapshot(token, preferred_home_id=None, timeout=WEBSOCKET_TIMEOUT_SECONDS):
    """Discover the endpoint/home and close after the first valid sample."""
    websocket_url, home = fetch_live_config(token, preferred_home_id)
    return asyncio.run(_receive_one_snapshot(websocket_url, home["id"], token, timeout))


def init_database(db_path=DEFAULT_DB):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS tibber_live_snapshots (
          timestamp TEXT PRIMARY KEY, timestamp_epoch REAL NOT NULL,
          power_w REAL, average_power_w REAL,
          min_power_w REAL, max_power_w REAL, accumulated_consumption_kwh REAL,
          accumulated_consumption_last_hour_kwh REAL, last_meter_consumption_kwh REAL,
          signal_strength REAL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tibber_consumption_intervals (
          start_timestamp TEXT NOT NULL, end_timestamp TEXT PRIMARY KEY,
          duration_seconds REAL NOT NULL, energy_kwh REAL, average_power_w REAL,
          slot_date TEXT, slot_index INTEGER, quality TEXT NOT NULL,
          reason TEXT NOT NULL, source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tibber_interval_slot
          ON tibber_consumption_intervals(slot_date, slot_index, quality);
        """)


def _snapshot_from_row(row):
    return LiveSnapshot(dt.datetime.fromisoformat(row[0]), *row[1:9])


def _nearest_quarter(value):
    local = value.astimezone(LOCAL_TZ)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (local - midnight).total_seconds()
    rounded = round(seconds / 900) * 900
    boundary = midnight + dt.timedelta(seconds=rounded)
    return boundary, abs((local - boundary).total_seconds())


def assign_quarter_slot(start, end):
    start_boundary, start_error = _nearest_quarter(start)
    end_boundary, end_error = _nearest_quarter(end)
    if start_error > QUARTER_BOUNDARY_TOLERANCE_SECONDS or end_error > QUARTER_BOUNDARY_TOLERANCE_SECONDS:
        return None
    if (end_boundary.astimezone(dt.timezone.utc) -
            start_boundary.astimezone(dt.timezone.utc)).total_seconds() != 900:
        return None
    return start_boundary.date().isoformat(), start_boundary.hour * 4 + start_boundary.minute // 15


def calculate_interval(previous, current):
    duration = (current.timestamp.astimezone(dt.timezone.utc) -
                previous.timestamp.astimezone(dt.timezone.utc)).total_seconds()
    result = {"start_timestamp": previous.timestamp.isoformat(),
              "end_timestamp": current.timestamp.isoformat(), "duration_seconds": duration,
              "energy_kwh": None, "average_power_w": None, "slot_date": None,
              "slot_index": None, "quality": "invalid", "reason": "invalid",
              "source": "pulse_snapshot"}
    if duration <= 0:
        result["reason"] = "timestamp not increasing"
        return result
    if duration < MIN_INTERVAL_SECONDS:
        result.update(quality="invalid", reason="interval too short")
    elif duration > MAX_INTERVAL_SECONDS:
        result.update(quality="gap", reason="interval too long")
    meter_delta = None
    if previous.last_meter_consumption_kwh is not None and current.last_meter_consumption_kwh is not None:
        candidate = current.last_meter_consumption_kwh - previous.last_meter_consumption_kwh
        if candidate >= 0:
            meter_delta = candidate
    energy = meter_delta
    counter = "lastMeterConsumption"
    if energy is None:
        counter = "accumulatedConsumption"
        if previous.accumulated_consumption_kwh is not None and current.accumulated_consumption_kwh is not None:
            candidate = current.accumulated_consumption_kwh - previous.accumulated_consumption_kwh
            if candidate >= 0:
                energy = candidate
            else:
                result.update(quality="counter_reset", reason="accumulatedConsumption reset")
    if energy is None:
        if result["quality"] != "counter_reset":
            result.update(quality="invalid", reason="counter missing or backwards")
        return result
    average = energy * 3_600_000 / duration
    result.update(energy_kwh=energy, average_power_w=average)
    if average > MAX_AVERAGE_POWER_W:
        result.update(quality="invalid", reason="implausible counter jump")
        return result
    if result["reason"] in ("interval too short", "interval too long"):
        return result
    slot = assign_quarter_slot(previous.timestamp, current.timestamp)
    if not slot:
        result.update(quality="invalid", reason="not near consecutive quarter boundaries")
        return result
    result.update(slot_date=slot[0], slot_index=slot[1], quality="valid_quarter",
                  reason=f"valid using {counter}")
    return result


def store_snapshot(snapshot, db_path=DEFAULT_DB):
    """Persist a sample and its predecessor interval; return (inserted, interval)."""
    init_database(db_path)
    with sqlite3.connect(db_path) as connection:
        previous_row = connection.execute("""SELECT timestamp, power_w, average_power_w,
          min_power_w, max_power_w, accumulated_consumption_kwh,
          accumulated_consumption_last_hour_kwh, last_meter_consumption_kwh, signal_strength
          FROM tibber_live_snapshots WHERE timestamp_epoch < ? ORDER BY timestamp_epoch DESC LIMIT 1""",
          (snapshot.timestamp.timestamp(),)).fetchone()
        values = asdict(snapshot)
        values["timestamp"] = snapshot.timestamp.isoformat()
        cursor = connection.execute("""INSERT OR IGNORE INTO tibber_live_snapshots
          (timestamp,timestamp_epoch,power_w,average_power_w,min_power_w,max_power_w,
           accumulated_consumption_kwh,accumulated_consumption_last_hour_kwh,
           last_meter_consumption_kwh,signal_strength,created_at) VALUES
          (:timestamp,:timestamp_epoch,:power_w,:average_power_w,:min_power_w,:max_power_w,
           :accumulated_consumption_kwh,:accumulated_consumption_last_hour_kwh,
           :last_meter_consumption_kwh,:signal_strength,:created_at)""",
          {**values, "timestamp_epoch": snapshot.timestamp.timestamp(),
           "created_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        if not cursor.rowcount:
            return False, None
        interval = calculate_interval(_snapshot_from_row(previous_row), snapshot) if previous_row else None
        if interval:
            connection.execute("""INSERT OR REPLACE INTO tibber_consumption_intervals VALUES
              (:start_timestamp,:end_timestamp,:duration_seconds,:energy_kwh,:average_power_w,
               :slot_date,:slot_index,:quality,:reason,:source)""", interval)
        return True, interval


def load_local_quarter_series(day, db_path=DEFAULT_DB):
    values = [None] * 96
    if not os.path.exists(db_path):
        return values
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("""SELECT slot_index, average_power_w
          FROM tibber_consumption_intervals WHERE slot_date=? AND quality='valid_quarter'""",
          (day.isoformat(),)).fetchall()
    for slot, watts in rows:
        if slot is not None and 0 <= slot < 96 and watts is not None:
            values[slot] = float(watts)
    return values


def status_text(db_path=DEFAULT_DB):
    init_database(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("""SELECT timestamp, power_w, average_power_w, min_power_w,
          max_power_w, accumulated_consumption_kwh, accumulated_consumption_last_hour_kwh,
          last_meter_consumption_kwh, signal_strength FROM tibber_live_snapshots
          ORDER BY timestamp_epoch DESC LIMIT 2""").fetchall()
    if not rows:
        return "No snapshots stored."
    current = _snapshot_from_row(rows[0])
    lines = [f"Last snapshot: {current.timestamp.isoformat(sep=' ')}",
             f"Power: {current.power_w:.0f} W" if current.power_w is not None else "Power: -- W",
             f"Meter: {current.last_meter_consumption_kwh:.4f} kWh" if current.last_meter_consumption_kwh is not None else "Meter: --"]
    if len(rows) > 1:
        previous = _snapshot_from_row(rows[1]); interval = calculate_interval(previous, current)
        lines += [f"Previous snapshot: {previous.timestamp.isoformat(sep=' ')}",
                  f"Energy: {interval['energy_kwh']:.4f} kWh" if interval['energy_kwh'] is not None else "Energy: --",
                  f"Duration: {interval['duration_seconds']:.0f} s",
                  f"Average: {interval['average_power_w']:.0f} W" if interval['average_power_w'] is not None else "Average: --",
                  f"Quarter usable: {'YES' if interval['quality'] == 'valid_quarter' else 'NO'}",
                  f"Reason: {interval['reason']}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    if args.status:
        print(status_text(args.db))
    else:
        parser.error("use --status (the display application fetches live data)")


if __name__ == "__main__":
    main()
