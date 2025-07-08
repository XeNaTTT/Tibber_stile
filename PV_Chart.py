#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import time
import datetime
import requests
import logging
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont

# Lokale Zeitzone
local_tz = ZoneInfo("Europe/Berlin")
# Pfade zum Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

logging.basicConfig(level=logging.DEBUG)

# DTU-SN und lokale Web-API (Docker im Host-Netzwerk)
DTU_SN   = "DTUBI-4143A019CB05"
API_BASE = "http://127.0.0.1:5000"


def get_historical_pv():
    """
    Versucht PV-Daten vom lokalen Docker-Service zu holen.
    Unterstuetzte Endpunkte: appGetHistPower, appGetPower
    Liefert je Liste fuer gestern und heute zurck.
    """
    endpoints = ["appGetHistPower", "getRealDataNew", "getRealDataHms", "getRealData"]
    data = None
    for ep in endpoints:
        url = f"{API_BASE}/{ep}/{DTU_SN}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            raw = r.json()
            raw_data = raw.get("data", raw)
            logging.debug(f"PV-Endpoint '{ep}' liefert Keys: {list(raw_data.keys())}")
            data = raw_data
            break
        except Exception as e:
            logging.warning(f"Endpoint '{ep}' failed: {e}")
    if not data:
        logging.error("Kein PV-Daten-Endpunkt verfgbar, liefere leere Listen.")
        return [], []

    # Leistung und Zeit erzeugen basierend auf powerArray und Schrittweite
    powers     = data.get("powerArray", [])
    abs_start  = data.get("absoluteStart", 0)
    step       = data.get("stepTime", 0)
    start_ts   = datetime.datetime.fromtimestamp(abs_start, tz=local_tz)

    records = []
    for i, p in enumerate(powers):
        ts = start_ts + datetime.timedelta(seconds=step * i)
        records.append({"startsAt": ts.isoformat(), "power": p})

    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    left  = [e for e in records if datetime.datetime.fromisoformat(e["startsAt"]).date() == yesterday]
    right = [e for e in records if datetime.datetime.fromisoformat(e["startsAt"]).date() == today]

    logging.debug(f"PV-Daten: gestern={len(left)} Punkte, heute={len(right)} Punkte")
    return left, right



def draw_pv_chart(d, left, right, fonts):
    # Falls keine PV-Daten vorliegen, Hinweis anzeigen
    if not left and not right:
        d.text((60, 80), "PV-Daten nicht verfgbar", font=fonts["small"], fill=0)
        return

    X0, X1 = 60, 800
    Y0, Y1 =   0, 160
    W, H    = X1 - X0, Y1 - Y0
    PW      = W / 2

    vals_l = [e["power"] for e in left]
    vals_r = [e["power"] for e in right]
    vmin   = 0
    vmax   = max(vals_l + vals_r + [1])
    sy     = H / (vmax - vmin)

    # Y-Achse
    step = max(vmax // 5, 1)
    yv = 0
    while yv <= vmax:
        y = Y1 - (yv - vmin) * sy
        d.line((X0, y, X1, y), fill=0)
        d.text((X0 - 45, y - 7), f"{int(yv)}W", font=fonts["small"], fill=0)
        yv += step
    d.text((X0, Y0 + 5), "PV-Erzeugung (W)", font=fonts["small"], fill=0)

    # Verlauf gestern
    if len(vals_l) > 1:
        xL = [X0 + i * (PW / (len(vals_l) - 1)) for i in range(len(vals_l))]
        for i in range(len(vals_l) - 1):
            d.line(
                (xL[i],   Y1 - (vals_l[i]   - vmin) * sy,
                 xL[i+1], Y1 - (vals_l[i+1] - vmin) * sy),
                fill=0, width=2
            )
    elif len(vals_l) == 1:
        x = X0 + PW / 2
        y = Y1 - (vals_l[0] - vmin) * sy
        d.ellipse((x-3, y-3, x+3, y+3), fill=0)

    # Verlauf heute
    if len(vals_r) > 1:
        xR = [X0 + PW + i * (PW / (len(vals_r) - 1)) for i in range(len(vals_r))]
        for i in range(len(vals_r) - 1):
            d.line(
                (xR[i],   Y1 - (vals_r[i]   - vmin) * sy,
                 xR[i+1], Y1 - (vals_r[i+1] - vmin) * sy),
                fill=0, width=2
            )
    elif len(vals_r) == 1:
        x = X0 + PW + PW / 2
        y = Y1 - (vals_r[0] - vmin) * sy
        d.ellipse((x-3, y-3, x+3, y+3), fill=0)

    bf = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
    )
    d.text((X0 + 10, Y1 + 5),     "Gestern", font=bf, fill=0)
    d.text((X0 + PW + 10, Y1 + 5),"Heute",   font=bf, fill=0)


def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = { "small": ImageFont.load_default() }

    left_pv, right_pv = get_historical_pv()
    draw_pv_chart(d, left_pv, right_pv, fonts)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()
