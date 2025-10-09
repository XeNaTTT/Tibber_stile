#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
import datetime as dt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math

# E-Paper-Bibliothek
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# API-Key
import api_key

# Display
EPD_WIDTH, EPD_HEIGHT = 800, 480

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
TZ = ZoneInfo("Europe/Berlin")

# Fonts
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
FONT_BOLD  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)

# ------------------------------
# 1. TIBBER: Preis + Verbrauch
# ------------------------------
def get_tibber_data():
    query = """
    {
      viewer {
        homes {
          currentSubscription {
            priceInfo {
              today { total startsAt }
              tomorrow { total startsAt }
              current { total startsAt }
            }
            current {
              power
              powerProduction
            }
          }
        }
      }
    }
    """
    headers = {
        "Authorization": f"Bearer {api_key.API_KEY}",
        "Content-Type": "application/json"
    }
    print("DEBUG – Header:", headers)          # ?
    res = requests.post(
        "https://api.tibber.com/v1-beta/gql",
        json={"query": query},
        headers=headers,
        timeout=15
    )
    print("DEBUG – Status:", res.status_code)  # ?
    print("DEBUG – Raw:", res.text)            # ?
    res.raise_for_status()
    payload = res.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    data = payload["data"]["viewer"]["homes"][0]["currentSubscription"]
    return data["priceInfo"], data["current"]
    
# ------------------------------
# 3. ECOFLOW: Batterie-Status
# ------------------------------
def get_ecoflow_status():
    # Platzhalter – ersetze durch echte API/MQTT
    return {
        "soc": 85,
        "power": -234,
        "time_remain": 142,
        "temp": 22
    }

# ------------------------------
# 4. MINI-GRAPH
# ------------------------------
def mini_graph(values, size=(200, 100), title=""):
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.plot(values, color="black", linewidth=2)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()).convert("1")
    plt.close(fig)
    return img

# ------------------------------
# 5. 15-MIN CHART
# ------------------------------
def draw_two_day_chart(draw, left, right, area, cur_dt, cur_price):
    X0, Y0, X1, Y1 = area
    W, H = X1 - X0, Y1 - Y0
    PW = W // 2

    def extract(slots):
        ts_list, val_list = [], []
        for s in slots:
            ts = dt.datetime.fromisoformat(s['startsAt']).astimezone(TZ)
            ts_list.append(ts)
            val_list.append(s['total'] * 100)
        return ts_list, val_list

    tl, vl = extract(left)
    tr, vr = extract(right)
    allp = vl + vr
    if not allp:
        return

    vmin, vmax = min(allp) - 0.5, max(allp) + 0.5
    sy = H / (vmax - vmin)

    # Y-Achse
    step = 5
    yv = math.floor(vmin / step) * step
    while yv <= vmax:
        yy = Y1 - (yv - vmin) * sy
        draw.line((X0 - 5, yy, X0, yy), fill=0)
        draw.line((X1, yy, X1 + 5, yy), fill=0)
        draw.text((X0 - 45, yy - 7), f"{yv / 100:.2f}", font=FONT_SMALL, fill=0)
        yv += step
    draw.text((X0 - 45, Y0 - 20), 'Preis (ct/kWh)', font=FONT_SMALL, fill=0)

    # Trenner
    draw.line((X0 + PW, Y0, X0 + PW, Y1), fill=0, width=2)

    def panel(ts_list, val_list, x0):
        n = len(val_list)
        if n < 2:
            return
        xs = [x0 + i * (PW / (n - 1)) for i in range(n)]

        # Linie
        for i in range(n - 1):
            x1, y1 = xs[i], Y1 - (val_list[i] - vmin) * sy
            x2, y2 = xs[i + 1], Y1 - (val_list[i + 1] - vmin) * sy
            draw.line((x1, y1, x2, y1), fill=0, width=2)
            draw.line((x2, y1, x2, y2), fill=0, width=2)

        # Min/Max Label
        for idx in (val_list.index(min(val_list)), val_list.index(max(val_list))):
            xi, yi = xs[idx], Y1 - (val_list[idx] - vmin) * sy
            draw.text((xi - 12, yi - 12), f"{val_list[idx] / 100:.2f}", font=FONT_SMALL, fill=0)

        # Aktueller Marker (minutengenau)
        if cur_dt and cur_price is not None:
            closest = min(range(n), key=lambda i: abs((ts_list[i] - cur_dt).total_seconds()))
            px = xs[closest]
            py = Y1 - (cur_price - vmin) * sy
            r = 4
            draw.ellipse((px - r, py - r, px + r, py + r), fill=0)
            draw.text((px + r + 2, py - r - 2), f"{cur_price / 100:.2f}", font=FONT_SMALL, fill=0)

    # Linkes Panel
    panel(tl, vl, X0)
    # Rechtes Panel
    panel(tr, vr, X0 + PW)

# ------------------------------
# 6. HAUPT-FUNKTION
# ------------------------------
def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    # Daten
    price_info, current_power = get_tibber_data()
    sunshine = get_weather()
    eco = get_ecoflow_status()

    # Bild
    img = Image.new("1", (EPD_WIDTH, EPD_HEIGHT), 255)
    draw = ImageDraw.Draw(img)

    # OBEN LINKS: Wetter
    x0, y0, w, h = 10, 10, 390, 230
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=0, width=2)
    draw.text((x0 + 10, y0 + 5), "Wetter & Sonnenstunden", font=FONT_BOLD, fill=0)
    draw.text((x0 + 10, y0 + 25), f"Heute: {int(sunshine[0]/60)} min", font=FONT_SMALL, fill=0)
    draw.text((x0 + 10, y0 + 45), f"Morgen: {int(sunshine[1]/60)} min", font=FONT_SMALL, fill=0)
    sun_graph = mini_graph([s/60 for s in sunshine], size=(370, 120), title="Sonnenstunden (min)")
    img.paste(sun_graph, (x0 + 10, y0 + 90))

    # OBEN RECHTS: EcoFlow
    x0, y0, w, h = 410, 10, 380, 230
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=0, width=2)
    draw.text((x0 + 10, y0 + 5), "EcoFlow Stream AC", font=FONT_BOLD, fill=0)
    draw.text((x0 + 10, y0 + 25), f"SOC: {eco['soc']}%", font=FONT_SMALL, fill=0)
    draw.text((x0 + 10, y0 + 45), f"Leistung: {eco['power']}W", font=FONT_SMALL, fill=0)
    draw.text((x0 + 10, y0 + 65), f"Restzeit: {eco['time_remain']} min", font=FONT_SMALL, fill=0)
    draw.text((x0 + 10, y0 + 85), f"Temperatur: {eco['temp']}°C", font=FONT_SMALL, fill=0)

    # UNTEN: Strompreis
    today = price_info["today"]
    tomorrow = price_info["tomorrow"] or []
    draw_two_day_chart(
        draw,
        left=today,
        right=tomorrow,
        area=(10, 250, 790, 470),
        cur_dt=dt.datetime.now(TZ),
        cur_price=price_info["current"]["total"] * 100
    )

    # Verbrauch
    draw.text((10, 430), f"Verbrauch: {current_power['power']}W | PV: {current_power['powerProduction']}W", font=FONT_SMALL, fill=0)

    # Update-Zeit
    footer = dt.datetime.now(TZ).strftime("Update: %H:%M %d.%m.%Y")
    draw.text((10, EPD_HEIGHT - 20), footer, font=FONT_SMALL, fill=0)

    # Display
    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()
