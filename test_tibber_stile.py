import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from energy_chart_utils import format_power_peak, price_slots_to_quarters

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


if __name__ == "__main__":
    unittest.main()
