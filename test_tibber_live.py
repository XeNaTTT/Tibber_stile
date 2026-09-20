import datetime as dt
import os
import tempfile
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from energy_chart_utils import merge_consumption_series
from tibber_live import (LiveSnapshot, assign_quarter_slot, calculate_interval,
                         load_local_quarter_series, parse_snapshot,
                         select_realtime_home, store_snapshot)

BERLIN = ZoneInfo("Europe/Berlin")


def snap(value, meter=None, accumulated=None):
    return LiveSnapshot(value, last_meter_consumption_kwh=meter,
                        accumulated_consumption_kwh=accumulated)


class SnapshotTests(unittest.TestCase):
    def test_parsing_and_null_fields(self):
        value = parse_snapshot({"timestamp": "2026-09-20T13:49:15+02:00",
                                "power": 153, "averagePower": None})
        self.assertEqual(153, value.power_w)
        self.assertIsNone(value.average_power_w)

    def test_home_selection_uses_feature_and_optional_id(self):
        homes = [{"id": "off", "features": {"realTimeConsumptionEnabled": False}},
                 {"id": "on", "features": {"realTimeConsumptionEnabled": True}}]
        self.assertEqual("on", select_realtime_home(homes)["id"])
        self.assertIsNone(select_realtime_home(homes, "off"))


class IntervalTests(unittest.TestCase):
    def test_meter_delta_and_actual_duration_power(self):
        result = calculate_interval(
            snap(dt.datetime(2026, 9, 20, 14, 0, 8, tzinfo=BERLIN), 10),
            snap(dt.datetime(2026, 9, 20, 14, 15, 12, tzinfo=BERLIN), 10.0412))
        self.assertEqual("valid_quarter", result["quality"])
        self.assertAlmostEqual(0.0412, result["energy_kwh"])
        self.assertAlmostEqual(0.0412 * 3_600_000 / 904, result["average_power_w"])
        self.assertEqual(56, result["slot_index"])

    def test_accumulated_fallback(self):
        result = calculate_interval(
            snap(dt.datetime(2026, 9, 20, 14, 0, tzinfo=BERLIN), accumulated=1),
            snap(dt.datetime(2026, 9, 20, 14, 15, tzinfo=BERLIN), accumulated=1.04))
        self.assertEqual("valid_quarter", result["quality"])
        self.assertAlmostEqual(.04, result["energy_kwh"])

    def test_day_reset_and_backwards_counter(self):
        result = calculate_interval(
            snap(dt.datetime(2026, 9, 20, 23, 45, tzinfo=BERLIN), accumulated=4),
            snap(dt.datetime(2026, 9, 21, 0, 0, tzinfo=BERLIN), accumulated=.01))
        self.assertEqual("counter_reset", result["quality"])
        result = calculate_interval(
            snap(dt.datetime(2026, 9, 20, 14, 0, tzinfo=BERLIN), 9),
            snap(dt.datetime(2026, 9, 20, 14, 15, tzinfo=BERLIN), 8))
        self.assertEqual("invalid", result["quality"])

    def test_valid_14_and_16_minutes_but_not_30(self):
        base = dt.datetime(2026, 9, 20, 14, 0, tzinfo=BERLIN)
        for minutes in (14, 16):
            result = calculate_interval(snap(base, 1), snap(base + dt.timedelta(minutes=minutes), 1.02))
            self.assertEqual("valid_quarter", result["quality"])
        result = calculate_interval(snap(base, 1), snap(base + dt.timedelta(minutes=30), 1.04))
        self.assertEqual("gap", result["quality"])

    def test_off_boundary_is_not_assigned(self):
        start = dt.datetime(2026, 9, 20, 14, 7, tzinfo=BERLIN)
        self.assertIsNone(assign_quarter_slot(start, start + dt.timedelta(minutes=15)))

    def test_midnight_slot(self):
        start = dt.datetime(2026, 9, 20, 23, 45, tzinfo=BERLIN)
        self.assertEqual(("2026-09-20", 95), assign_quarter_slot(start, start + dt.timedelta(minutes=15)))

    def test_dst_uses_real_elapsed_time(self):
        start = dt.datetime(2026, 10, 25, 2, 45, tzinfo=BERLIN, fold=0)
        end = dt.datetime(2026, 10, 25, 2, 0, tzinfo=BERLIN, fold=1)
        self.assertEqual(900, calculate_interval(snap(start, 1), snap(end, 1.01))["duration_seconds"])


class PersistenceAndChartTests(unittest.TestCase):
    def test_database_duplicate_persistence_and_local_series(self):
        with tempfile.TemporaryDirectory() as directory:
            db = os.path.join(directory, "pulse.db")
            first = snap(dt.datetime(2026, 9, 20, 14, 0, tzinfo=BERLIN), 1)
            second = snap(dt.datetime(2026, 9, 20, 14, 15, tzinfo=BERLIN), 1.04)
            self.assertTrue(store_snapshot(first, db)[0])
            self.assertFalse(store_snapshot(first, db)[0])
            self.assertEqual("valid_quarter", store_snapshot(second, db)[1]["quality"])
            self.assertIsNotNone(load_local_quarter_series(first.timestamp.date(), db)[56])

    def test_historical_fallback_and_local_priority(self):
        historical = [None] * 96; historical[40] = 400; historical[56] = 300
        local = [None] * 96; local[56] = 164
        merged = merge_consumption_series(historical, local)
        self.assertEqual(400, merged[40])
        self.assertEqual(164, merged[56])
        self.assertIsNone(merged[41])

    @mock.patch("tibber_live.fetch_live_config", return_value=("wss://example.invalid", {"id": "home"}))
    @mock.patch("tibber_live.asyncio.run")
    def test_live_measurement_timeout(self, run, _config):
        from tibber_live import get_tibber_live_snapshot
        def timeout(coroutine):
            coroutine.close()
            raise TimeoutError
        run.side_effect = timeout
        with self.assertRaises(TimeoutError):
            get_tibber_live_snapshot("secret")


if __name__ == "__main__":
    unittest.main()
