import datetime as dt
import sys
import types
import unittest

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - production dependency may be absent in CI
    Image = ImageDraw = ImageFont = None


# The production renderer imports its Raspberry Pi driver at module load time.
if Image is not None:
    driver_package = types.ModuleType("waveshare_epd")
    driver_package.epd7in5_V2 = types.SimpleNamespace()
    sys.modules.setdefault("waveshare_epd", driver_package)
    import Tibber_stile as dashboard


@unittest.skipIf(Image is None, "Pillow is not installed")
class CurrentWeatherTests(unittest.TestCase):
    def test_current_weather_fields_are_normalized(self):
        current = dashboard._parse_current_weather({"current": {
            "time": "2026-09-20T20:45", "temperature_2m": 17.2,
            "weather_code": 61, "relative_humidity_2m": 88,
            "wind_speed_10m": 22.1, "wind_direction_10m": 225,
            "is_day": 0,
        }})
        self.assertEqual(17.2, current["temperature"])
        self.assertEqual(61, current["code"])
        self.assertEqual(88.0, current["relative_humidity"])
        self.assertEqual(22.1, current["wind_speed"])
        self.assertEqual(225.0, current["wind_direction"])
        self.assertFalse(current["is_day"])

    def test_wmo_text_and_wind_direction(self):
        self.assertEqual("Regen", dashboard.weather_code_text(61))
        self.assertEqual("Schneeschauer", dashboard.weather_code_text(86))
        expected = ("Nord", "Nordost", "Ost", "Suedost", "Sued",
                    "Suedwest", "West", "Nordwest")
        self.assertEqual(expected, tuple(
            dashboard.wind_direction_text(value) for value in range(0, 360, 45)
        ))

    def test_missing_current_data_still_renders_inside_display(self):
        image = Image.new("1", (dashboard.DISPLAY_WIDTH,
                                dashboard.DISPLAY_HEIGHT), 1)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        fonts = {key: font for key in (
            "panel_bold", "panel_small", "panel_tiny", "panel_temperature",
            "panel_condition",
        )}
        dashboard.draw_current_weather_panel(
            draw, image,
            (dashboard.WEATHER_PANEL_X, 0, dashboard.DISPLAY_WIDTH,
             dashboard.DISPLAY_HEIGHT),
            fonts, None,
        )
        self.assertEqual((800, 480), image.size)
        self.assertEqual(176, dashboard.WEATHER_PANEL_WIDTH)
        self.assertIsNotNone(image.getbbox())


if __name__ == "__main__":
    unittest.main()
