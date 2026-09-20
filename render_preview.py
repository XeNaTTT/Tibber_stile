#!/usr/bin/env python3
"""Render the weather dashboard and all supplied Image2Lcd assets locally."""

import argparse
import sys
import types

from PIL import Image, ImageDraw, ImageFont

# The production module imports the Raspberry Pi display driver at module load
# time.  A preview only needs its pure rendering functions.
if "waveshare_epd" not in sys.modules:
    driver_package = types.ModuleType("waveshare_epd")
    driver_package.epd7in5_V2 = types.SimpleNamespace()
    sys.modules["waveshare_epd"] = driver_package
for optional_dependency in ("requests", "pandas", "numpy"):
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
        draw, image, 10, 10, 780, 164, fonts(), weather_days, 6.4, 3.8
    )
    draw.text((10, 200), "Wetter-Preview (native 90 x 81 px, ohne Dithering)",
              font=fonts()["bold"], fill=0)
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
