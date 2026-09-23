import datetime as dt
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

if Image is not None:
    driver_package = types.ModuleType("waveshare_epd")
    driver_package.epd7in5_V2 = types.SimpleNamespace()
    sys.modules.setdefault("waveshare_epd", driver_package)
    import Tibber_stile as dashboard


@unittest.skipIf(Image is None, "Pillow is not installed")
class CurrentWeatherTests(unittest.TestCase):
    def test_complete_wmo_icon_mapping(self):
        cases = (
            ((0, True), "sonnig.c"), ((0, False), "leicht_bewoelkt_nacht.c"),
            ((1, True), "leicht_bewoelkt.c"), ((1, False), "leicht_bewoelkt_nacht.c"),
            ((2, True), "Wolkig.c"), ((2, False), "wolkig_nachts.c"),
            ((3, True), "Wolkig.c"), ((3, False), "bewoelkt_nacht.c"),
            ((45, True), "Nebel.c"), ((51, True), "niesel.c"),
            ((61, True), "regen.c"), ((71, True), "regen.c"),
            ((77, False), "niesel.c"), ((95, True), "gewitter.c"),
            ((95, False), "gewitter_nacht.c"),
            ((1234, True), "Wolkig.c"), ((1234, False), "wolkig_nachts.c"),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(expected, dashboard.resolve_weather_icon(*arguments))

    def test_wind_only_overrides_dry_codes(self):
        self.assertEqual("windig.c", dashboard.resolve_weather_icon(1, True, 35))
        self.assertEqual("gewitter.c", dashboard.resolve_weather_icon(95, True, 80))
        self.assertEqual("regen.c", dashboard.resolve_weather_icon(61, True, 80))
        self.assertEqual("Nebel.c", dashboard.resolve_weather_icon(45, True, 80))

    def test_all_twelve_assets_parse(self):
        self.assertEqual(12, len(dashboard.WEATHER_ICON_FILES))
        for filename in dashboard.WEATHER_ICON_FILES.values():
            with self.subTest(filename=filename):
                data, width, height, _ = dashboard.load_c_bitmap(
                    os.path.join(dashboard.WEATHER_ICON_DIR, filename))
                self.assertEqual((220, 220), (width, height))
                self.assertTrue(data)

    def test_missing_icon_falls_back_then_renders_nothing(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(dashboard, "WEATHER_ICON_DIR", directory):
            self.assertIsNone(dashboard._get_weather_icon_image(0, True))
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(dashboard, "WEATHER_ICON_DIR", directory), \
             mock.patch.dict(dashboard.WEATHER_ICON_FILES,
                             {"sonnig": "missing.c", "wolkig": "Wolkig.c"}):
            source = os.path.join(os.path.dirname(__file__), "Wolkig.c")
            with open(source, "rb") as src, open(os.path.join(directory, "Wolkig.c"), "wb") as dst:
                dst.write(src.read())
            self.assertIsNotNone(dashboard._get_weather_icon_image(0, True))

    def test_aspect_ratio_preserving_scaling(self):
        source = Image.new("1", (200, 100), 1)
        self.assertEqual((120, 60), dashboard.fit_weather_icon(source, 120, 120).size)
        self.assertEqual((80, 40), dashboard.fit_weather_icon(source, 100, 40).size)

    def test_current_weather_fields_are_normalized(self):
        current = dashboard._parse_current_weather({"current": {
            "time": "2026-09-20T20:45", "temperature_2m": 17.2,
            "weather_code": 61, "relative_humidity_2m": 88,
            "wind_speed_10m": 22.1, "wind_direction_10m": 225, "is_day": 0,
        }})
        self.assertEqual((61, False), (current["code"], current["is_day"]))

    def test_current_panel_renders_on_white_without_data(self):
        image = Image.new("1", (dashboard.DISPLAY_WIDTH, dashboard.DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        fonts = {key: font for key in ("panel_bold", "panel_small", "panel_tiny",
                                       "panel_temperature", "panel_condition")}
        dashboard.draw_current_weather_panel(
            draw, image, (dashboard.WEATHER_PANEL_X, 0, dashboard.DISPLAY_WIDTH,
                          dashboard.DISPLAY_HEIGHT), fonts, None)
        self.assertEqual(1, image.getpixel((dashboard.WEATHER_PANEL_X, 479)))


if __name__ == "__main__":
    unittest.main()
