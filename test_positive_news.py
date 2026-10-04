import datetime as dt
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

import positive_news as news

driver_package = types.ModuleType("waveshare_epd")
driver_package.epd7in5_V2 = types.SimpleNamespace()
sys.modules.setdefault("waveshare_epd", driver_package)
import Tibber_stile as dashboard

RSS = b"""<?xml version='1.0' encoding='UTF-8'?><rss version='2.0'><channel><item>
<title>Forschung &amp; Fortschritt f\xc3\xbcr alle</title><link>https://example.test/story?utm_source=rss</link>
<description><![CDATA[<p>Eine neue L\xc3\xb6sung wurde entwickelt.</p> Weiterlesen]]></description>
<pubDate>Sun, 04 Oct 2026 08:00:00 +0000</pubDate></item>
<item><description>Ohne Titel</description></item></channel></rss>"""


def article(title, summary="", url="", published=None):
    return {"title": title, "summary": summary, "url": url,
            "source": "Test", "published": published}


class NewsLogicTests(unittest.TestCase):
    def test_rss_parsing_html_entities_unicode_and_unknown_item(self):
        items = news.parse_rss(RSS, "tagesschau")
        self.assertEqual(1, len(items))
        self.assertEqual("Forschung & Fortschritt für alle", items[0]["title"])
        self.assertEqual("Eine neue Lösung wurde entwickelt.", items[0]["summary"])

    def test_html_cleanup_removes_title_tags_and_boilerplate(self):
        self.assertEqual("Ärzte helfen.", news.clean_text(
            "Titel <b>Titel</b>: &Auml;rzte helfen. <br> mehr", "Titel Titel"))

    def test_malformed_xml_raises_for_independent_source_handling(self):
        with self.assertRaises(Exception):
            news.parse_rss(b"<rss><broken>", "bad")

    def test_deduplication_by_url_and_normalized_headline(self):
        values = [article("Erste Meldung", url="https://EXAMPLE.test/a?x=1"),
                  article("Andere Meldung", url="https://example.test/a"),
                  article("Neue Lösung!", url="https://example.test/b"),
                  article("  neue   lösung ", url="https://example.test/c")]
        self.assertEqual(2, len(news.deduplicate(values)))

    def test_positive_negative_and_constructive_medical_scoring(self):
        self.assertGreater(news.constructive_score(
            article("Durchbruch: Erneuerbare Lösung gelingt")), 8)
        self.assertLess(news.constructive_score(
            article("Angriff mit Toten nach Bombenexplosion")), -10)
        self.assertGreater(news.constructive_score(
            article("Neue Therapie verbessert Überlebenschancen bei Krebs")), 0)

    def test_recency_scoring_bands(self):
        now = dt.datetime(2026, 10, 4, 12, tzinfo=dt.timezone.utc)
        self.assertGreater(news.recency_score(now - dt.timedelta(hours=5), now),
                           news.recency_score(now - dt.timedelta(hours=30), now))

    def test_selection_maximum_three_and_topic_diversity(self):
        values = [article("Therapie verbessert Gesundheit", "Forschung und Behandlung"),
                  article("Neue Solar Lösung", "Klimaschutz und erneuerbare Energie"),
                  article("Digitale Innovation gelingt", "Neue Technologie"),
                  article("Zweite Therapie erfolgreich", "Medizinische Behandlung")]
        selected = news.select_stories(values)
        self.assertEqual(3, len(selected))
        self.assertEqual(3, len({item["topic"] for item in selected}))

    def test_maximum_three_source_sentences(self):
        self.assertEqual(3, len(news.split_sentences("Eins. Zwei! Drei? Vier.")))


class CacheAndNetworkTests(unittest.TestCase):
    class Response:
        def __init__(self, broken=False): self.content, self.broken = RSS, broken
        def raise_for_status(self):
            if self.broken: raise RuntimeError("HTTP 500")

    class Session:
        def get(self, url, **_kwargs):
            return CacheAndNetworkTests.Response("broken" in url)

    def test_atomic_cache_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.json")
            values = [article("Lösung gelingt")]
            news.write_cache(values, path)
            self.assertEqual(values, news.load_cache(path))
            self.assertEqual(["cache.json"], os.listdir(directory))
            with open(path, encoding="utf-8") as handle:
                self.assertIn("updated_at", json.load(handle))

    def test_rss_failure_with_and_without_cache_and_malformed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.json")
            cached = [article("Erfolgreiche Forschung")]
            news.write_cache(cached, path)
            result = news.refresh_positive_news(path, (("bad", "https://broken/rss"),), self.Session())
            self.assertEqual(cached, result)
            with open(path, "w") as handle: handle.write("not json")
            self.assertEqual([], news.refresh_positive_news(
                path, (("bad", "https://broken/rss"),), self.Session()))

    def test_one_broken_source_while_other_source_works(self):
        with tempfile.TemporaryDirectory() as directory:
            result = news.refresh_positive_news(os.path.join(directory, "cache.json"),
                (("bad", "https://broken/rss"), ("good", "https://working/rss")), self.Session())
            self.assertEqual(1, len(result))

    def test_interrupted_atomic_replace_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cache.json")
            original = [article("Bestehende Lösung")]
            news.write_cache(original, path)
            with mock.patch("positive_news.os.replace", side_effect=OSError("power loss")):
                with self.assertRaises(OSError): news.write_cache([article("Neu")], path)
            self.assertEqual(original, news.load_cache(path))


class PixelAndPanelTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("1", (dashboard.DISPLAY_WIDTH, dashboard.DISPLAY_HEIGHT), 1)
        self.draw = ImageDraw.Draw(self.image)
        base = "/usr/share/fonts/truetype/dejavu/"
        regular = ImageFont.truetype(base + "DejaVuSans.ttf", 10)
        bold = ImageFont.truetype(base + "DejaVuSans-Bold.ttf", 10)
        self.fonts = {key: regular for key in ("panel_small", "panel_tiny", "panel_condition",
                                                "panel_temperature", "panel_details", "news_body",
                                                "news_source")}
        self.fonts.update(panel_bold=bold, panel_compact_temperature=bold,
                          news_section=bold, news_headline=bold)

    def test_headline_body_pixel_wrapping_and_word_truncation(self):
        lines = dashboard.wrap_text_to_width(self.draw, "WWW iii Äpfel Lösung",
                                             self.fonts["news_body"], 55)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(self.draw.textbbox((0, 0), line,
                            font=self.fonts["news_body"])[2] <= 55 for line in lines))
        value = dashboard.truncate_to_width(self.draw,
            "Eine sehr lange deutsche Überschrift", self.fonts["news_body"], 100)
        self.assertTrue(value.endswith("..."))
        self.assertLessEqual(self.draw.textbbox((0, 0), value,
                             font=self.fonts["news_body"])[2], 100)

    def test_panel_renders_icon_wind_speed_news_but_no_direction(self):
        stories = [article(f"Headline {i} Lösung", "Ein Fortschritt gelingt.") for i in range(3)]
        calls = []
        original_text = self.draw.text
        def record(xy, text, *args, **kwargs):
            calls.append(str(text)); return original_text(xy, text, *args, **kwargs)
        with mock.patch.object(self.draw, "text", side_effect=record), \
             mock.patch.object(dashboard, "_get_weather_icon_image",
                               return_value=Image.new("1", (40, 40), 0)) as icon:
            dashboard.draw_current_weather_panel(self.draw, self.image,
                (dashboard.WEATHER_PANEL_X, 0, 800, 480), self.fonts,
                {"temperature": 15, "code": 1, "is_day": True,
                 "relative_humidity": 60, "wind_speed": 14,
                 "wind_direction": 225}, stories)
        rendered = " ".join(calls)
        self.assertIn("Wind 14 km/h", rendered)
        self.assertNotIn("Suedwest", rendered)
        self.assertIn("GUTE NACHRICHTEN", rendered)
        self.assertEqual(3, sum(value.startswith("Headline") for value in calls))
        icon.assert_called_once()

    def test_news_failure_fallback_still_renders_panel(self):
        dashboard.draw_current_weather_panel(self.draw, self.image,
            (dashboard.WEATHER_PANEL_X, 0, 800, 480), self.fonts, None, [])
        self.assertEqual(1, self.image.getpixel((799, 479)))


if __name__ == "__main__":
    unittest.main()
