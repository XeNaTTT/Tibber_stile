import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from energy_chart_utils import (
    consumption_to_quarter_series,
    format_power_peak,
    price_slots_to_quarters,
    quarter_slot_index,
    x_for_quarter_slot,
)

LOCAL_TZ = ZoneInfo("Europe/Berlin")


class PriceIntervalTests(unittest.TestCase):
    def _slot(self, hour, minute, price):
        timestamp = dt.datetime(2026, 9, 19, hour, minute, tzinfo=LOCAL_TZ)
        return {"startsAt": timestamp.isoformat(), "total": price}

    def test_hourly_prices_are_expanded_to_quarters(self):
        timestamps, prices = price_slots_to_quarters([
            self._slot(0, 0, 0.20), self._slot(1, 0, 0.24)
        ], LOCAL_TZ)
        self.assertEqual(8, len(timestamps))
        self.assertTrue(all(right - left == dt.timedelta(minutes=15)
                            for left, right in zip(timestamps, timestamps[1:])))
        self.assertEqual([20.0] * 4 + [24.0] * 4, prices)

    def test_quarter_prices_are_not_expanded_again(self):
        slots = [self._slot(0, minute, 0.20 + minute / 1000)
                 for minute in (0, 15, 30, 45)]
        timestamps, _ = price_slots_to_quarters(slots, LOCAL_TZ)
        self.assertEqual(4, len(timestamps))
        self.assertTrue(all(right - left == dt.timedelta(minutes=15)
                            for left, right in zip(timestamps, timestamps[1:])))

    def test_missing_quarter_is_filled_with_previous_price(self):
        timestamps, prices = price_slots_to_quarters([
            self._slot(0, 0, 0.20), self._slot(0, 30, 0.30),
            self._slot(0, 45, 0.40)
        ], LOCAL_TZ)
        self.assertEqual([0, 15, 30, 45], [timestamp.minute for timestamp in timestamps])
        self.assertEqual([20.0, 20.0, 30.0, 40.0], prices)


class PeakLabelTests(unittest.TestCase):
    def test_peak_is_formatted_in_watts_or_kilowatts(self):
        self.assertEqual("850 W", format_power_peak(850))
        self.assertEqual("1.25 kW", format_power_peak(1250))


class ConsumptionIntervalTests(unittest.TestCase):
    def test_quarter_consumption_uses_timestamp_slot_and_watts(self):
        day = dt.date(2026, 9, 19)
        nodes = [
            {"from": dt.datetime(2026, 9, 19, 9, 30, tzinfo=LOCAL_TZ).isoformat(),
             "consumption": 0.125},
        ]
        timestamps, values = consumption_to_quarter_series(nodes, day, LOCAL_TZ, "15min")
        self.assertEqual(96, len(values))
        self.assertEqual(38, quarter_slot_index(timestamps[38]))
        self.assertEqual(500.0, values[38])
        self.assertEqual(1, sum(value is not None for value in values))

    def test_hourly_consumption_is_not_upsampled(self):
        day = dt.date(2026, 9, 19)
        nodes = [
            {"from": dt.datetime(2026, 9, 19, 9, 0, tzinfo=LOCAL_TZ).isoformat(),
             "consumption": 0.4},
            {"from": dt.datetime(2026, 9, 19, 10, 0, tzinfo=LOCAL_TZ).isoformat(),
             "consumption": 0.8},
        ]
        _, values = consumption_to_quarter_series(nodes, day, LOCAL_TZ, "hourly")
        self.assertEqual(400.0, values[36])
        self.assertEqual(800.0, values[40])
        self.assertIsNone(values[37])
        self.assertEqual(2, sum(value is not None for value in values))

    def test_shared_x_scale_has_exact_day_boundary(self):
        self.assertEqual(100.0, x_for_quarter_slot(100, 384, 0))
        self.assertEqual(252.0, x_for_quarter_slot(100, 384, 38))
        self.assertEqual(484.0, x_for_quarter_slot(100, 384, 96))


if __name__ == "__main__":
    unittest.main()
