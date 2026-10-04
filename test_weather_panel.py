import datetime as dt
import os
import sys
import json
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
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def _fonts(self):
        font = ImageFont.load_default()
        return {key: font for key in ("panel_bold", "panel_small", "panel_tiny",
                                       "panel_temperature", "panel_condition")}

    def test_single_and_twopart_joke_parsing_with_unicode(self):
        self.assertEqual(
            ("Fröhliche Grüße aus Köln!", 12),
            dashboard.parse_joke_response({
                "type": "single", "joke": "Fröhliche Grüße aus Köln!", "id": 12,
            }),
        )
        self.assertEqual(
            ("Was macht ein Keks? Krümel.", 13),
            dashboard.parse_joke_response({
                "type": "twopart", "setup": "Was macht ein Keks?",
                "delivery": "Krümel.", "id": 13,
            }),
        )

    def test_pixel_word_wrapping_and_oversized_rejection(self):
        draw = ImageDraw.Draw(Image.new("1", (200, 100), 1))
        font = ImageFont.load_default()
        lines = dashboard.wrap_text_to_width(
            draw, "eins zwei drei vier", font,
            draw.textbbox((0, 0), "eins zwei", font=font)[2],
        )
        self.assertEqual("eins zwei drei vier".split(), " ".join(lines).split())
        self.assertGreater(len(lines), 1)
        self.assertFalse(dashboard.joke_fits(draw, "Untrennbarlangeswort", font, 5, 50))

    def test_maximum_length_and_recent_id_are_rejected(self):
        long_joke = "x " * dashboard.JOKE_MAX_CHARS
        responses = [
            self.FakeResponse({"type": "single", "joke": long_joke, "id": 1}),
            self.FakeResponse({"type": "single", "joke": "Doppelt.", "id": 2}),
            self.FakeResponse({"type": "single", "joke": "Neu und kurz.", "id": 3}),
        ]
        session = mock.Mock()
        session.get.side_effect = responses
        with tempfile.TemporaryDirectory() as directory:
            cache = os.path.join(directory, "joke.json")
            with open(cache, "w", encoding="utf-8") as handle:
                json.dump({"text": "Alt.", "recent_ids": [2]}, handle)
            self.assertEqual("Neu und kurz.", dashboard.get_random_short_joke(
                cache, session=session, max_attempts=3))
            self.assertEqual(3, session.get.call_count)

    def test_api_failure_uses_cache_or_blank(self):
        session = mock.Mock()
        session.get.side_effect = RuntimeError("offline")
        with tempfile.TemporaryDirectory() as directory:
            cache = os.path.join(directory, "joke.json")
            self.assertEqual("", dashboard.get_random_short_joke(cache, session=session))
            with open(cache, "w", encoding="utf-8") as handle:
                json.dump({"text": "Zwischengespeichert."}, handle)
            self.assertEqual("Zwischengespeichert.",
                             dashboard.get_random_short_joke(cache, session=session))

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

    def test_wind_direction_is_not_drawn_but_wind_speed_is(self):
        base = {
            "temperature": 17, "code": 0, "relative_humidity": 50,
            "wind_speed": 22, "is_day": True,
        }
        images = []
        for direction in (0, 225):
            image = Image.new("1", (dashboard.DISPLAY_WIDTH,
                                     dashboard.DISPLAY_HEIGHT), 1)
            weather = dict(base, wind_direction=direction)
            with mock.patch.object(dashboard, "_get_weather_icon_image", return_value=None):
                dashboard.draw_current_weather_panel(
                    ImageDraw.Draw(image), image,
                    (dashboard.WEATHER_PANEL_X, 0, dashboard.DISPLAY_WIDTH,
                     dashboard.DISPLAY_HEIGHT), self._fonts(), weather,
                    joke_text="Ein kurzer Witz.")
            images.append(image)
        self.assertEqual(images[0].tobytes(), images[1].tobytes())
        changed_speed = Image.new("1", (dashboard.DISPLAY_WIDTH,
                                         dashboard.DISPLAY_HEIGHT), 1)
        with mock.patch.object(dashboard, "_get_weather_icon_image", return_value=None):
            dashboard.draw_current_weather_panel(
                ImageDraw.Draw(changed_speed), changed_speed,
                (dashboard.WEATHER_PANEL_X, 0, dashboard.DISPLAY_WIDTH,
                 dashboard.DISPLAY_HEIGHT), self._fonts(),
                dict(base, wind_speed=23, wind_direction=0),
                joke_text="Ein kurzer Witz.")
        self.assertNotEqual(images[0].tobytes(), changed_speed.tobytes())


if __name__ == "__main__":
    unittest.main()
