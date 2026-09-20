#!/usr/bin/env python3
"""Render the weather dashboard and all supplied Image2Lcd assets locally."""

import argparse
import datetime as dt
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# The production module imports the Raspberry Pi display driver at module load
# time.  A preview only needs its pure rendering functions.
if "waveshare_epd" not in sys.modules:
    driver_package = types.ModuleType("waveshare_epd")
    driver_package.epd7in5_V2 = types.SimpleNamespace()
    sys.modules["waveshare_epd"] = driver_package
for optional_dependency in ("requests",):
    sys.modules.setdefault(optional_dependency, types.ModuleType(optional_dependency))

import Tibber_stile as dashboard


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def fonts():
    base = "/usr/share/fonts/truetype/dejavu/"
    return {
        "bold": load_font(base + "DejaVuSans-Bold.ttf", 14),
        "small": load_font(base + "DejaVuSans.ttf", 12),
        "tiny": load_font(base + "DejaVuSans.ttf", 10),
        "temperature": load_font(base + "DejaVuSans-Bold.ttf", 19),
        "panel_bold": load_font(base + "DejaVuSans-Bold.ttf", 16),
        "panel_small": load_font(base + "DejaVuSans.ttf", 12),
        "panel_tiny": load_font(base + "DejaVuSans.ttf", 11),
        "panel_temperature": load_font(base + "DejaVuSans.ttf", 67),
        "panel_condition": load_font(base + "DejaVuSans.ttf", 18),
    }


def render_dashboard(path):
    image = Image.new("1", (800, 480), 1)
    draw = ImageDraw.Draw(image)
    codes = ((0, 2, 61, 0), (3, 45, 80, 95))
    weather_days = []
    for day_index, day_codes in enumerate(codes):
        weather_days.append([
            {
                "temperature": (12, 17, 14, 8)[period] + day_index,
                "precipitation_probability": (5, 25, 70, 35)[period],
                "code": code,
                "is_day": period != 3,
            }
            for period, code in enumerate(day_codes)
        ])
    dashboard.draw_weather_dashboard(
        draw, image, 10, 10, dashboard.MAIN_CONTENT_WIDTH - 20, 164,
        fonts(), weather_days, 6.4, 3.8
    )
    preview_fonts = fonts()
    info = {
        "current_price": 28.4,
        "lowest_today": 17.2,
        "lowest_today_time": dt.datetime(2026, 9, 20, 3, 0),
        "highest_today": 42.8,
    }
    dashboard.draw_info_box(
        draw, info, preview_fonts, y=192,
        width=dashboard.MAIN_CONTENT_WIDTH - 20,
    )

    day = dt.datetime.now(dashboard.LOCAL_TZ).replace(hour=0, minute=0, second=0,
                                                       microsecond=0)
    prices = []
    tomorrow = []
    for slot in range(96):
        stamp = day + dt.timedelta(minutes=15 * slot)
        price = 0.25 + 0.09 * np.sin((slot - 22) * np.pi / 48)
        prices.append({"startsAt": stamp.isoformat(), "total": price})
        tomorrow.append({"startsAt": (stamp + dt.timedelta(days=1)).isoformat(),
                         "total": price + 0.025})
    consumption = [None] * 96
    for slot, value in zip(range(36, 65),
                           [310, 280, 260, 245, 230, 220, 240, 275, 330, 410,
                            520, 680, 890, 760, 610, 520, 470, 430, 460, 510,
                            590, 720, 980, 810, 690, 570, 500, 455, 420]):
        consumption[slot] = value
    pv = [max(0, 900 * np.sin((slot - 24) * np.pi / 56)) for slot in range(96)]
    sample = types.SimpleNamespace(power_w=134,
                                   accumulated_consumption_last_hour_kwh=0.12)
    dashboard.draw_two_day_chart(
        image, draw, prices, tomorrow, preview_fonts, ("Heute", "Morgen"),
        (10, 222, dashboard.MAIN_CONTENT_WIDTH - 10, 410),
        pv_left={"pv_sum": pd.Series(pv)},
        pv_right={"pv_sum": pd.Series(pv)},
        cons_left=pd.Series(consumption, dtype="float64"),
        cons_right=pd.Series(consumption, dtype="float64"),
        live_snapshot=sample,
    )
    current_weather = {
        "time": dt.datetime(2026, 9, 20, 20, 45, tzinfo=dashboard.LOCAL_TZ),
        "temperature": 17,
        "code": 61,
        "relative_humidity": 88,
        "wind_speed": 22,
        "wind_direction": 225,
        "is_day": False,
    }
    dashboard.draw_current_weather_panel(
        draw, image,
        (dashboard.WEATHER_PANEL_X, 0, dashboard.DISPLAY_WIDTH,
         dashboard.DISPLAY_HEIGHT),
        preview_fonts, current_weather,
    )
    draw.text((10, 470), "Update: Preview", font=preview_fonts["tiny"], fill=0)
    image.save(path)


def render_asset_sheet(path):
    items = list(dashboard.WEATHER_ICON_FILES.items())
    image = Image.new("1", (90 * len(items), 110), 1)
    draw = ImageDraw.Draw(image)
    font = fonts()["tiny"]
    for index, (name, filename) in enumerate(items):
        bitmap = dashboard._get_weather_icon_image(
            0 if name.startswith("clear") else {
                "partly": 2, "overcast": 3, "fog": 45, "rain": 61,
                "showers": 80, "thunder": 95, "snow": 71,
            }[name],
            name != "clear_night",
            invert=dashboard.ICON_INVERT,
            bitreverse=dashboard.ICON_BITREVERSE,
        )
        image.paste(bitmap, (index * 90, 0))
        draw.text((index * 90 + 2, 86), filename.removesuffix("_new.c"),
                  font=font, fill=0)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="screen_sim.png")
    parser.add_argument("--asset-out", default="weather_assets_preview.png")
    args = parser.parse_args()
    render_dashboard(args.out)
    render_asset_sheet(args.asset_out)
    print(f"Saved: {args.out}")
    print(f"Saved: {args.asset_out}")


if __name__ == "__main__":
    main()
