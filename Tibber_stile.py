#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
import time
import requests
import datetime
from PIL import Image, ImageDraw, ImageFont
from zoneinfo import ZoneInfo

# Lokale Zeitzone
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare-Treiber (auf deinem Pi)
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# DTU-SN und lokaler API-Endpoint (IPv4)
DTU_SN   = "DTUBI-4143A019CB05"
API_BASE = "http://127.0.0.1:5000"

def get_historical_pv():
    """
    Ruft die 2-Tage-Historie in Watt von der lokalen Web-API ab
    und gibt zwei Listen zurück: yesterday_data, today_data
    im Format [{"startsAt": ISO8601, "power": int}, ...]
    """
    url = f"{API_BASE}/appGetHistPower/{DTU_SN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    raw = r.json()
    # raw enthält z.B.:
    # { "power":[..], "time":[{"seconds":..,"nanos":..}, ...], ... }
    powers = raw.get("power", [])
    times  = raw.get("time", [])
    data = []
    for p, t in zip(powers, times):
        # Timestamp zusammenbauen
        ts = datetime.datetime.fromtimestamp(t.get("seconds", 0), tz=local_tz)
        nanos = t.get("nanos", 0)
        if nanos:
            ts += datetime.timedelta(microseconds=nanos/1000)
        data.append({"startsAt": ts.isoformat(), "power": p})
    # Aufteilen in Gestern und Heute
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    left  = [e for e in data if datetime.datetime.fromisoformat(e["startsAt"]).date() == yesterday]
    right = [e for e in data if datetime.datetime.fromisoformat(e["startsAt"]).date() == today]
    return left, right

def draw_pv_chart(d, left_data, right_data, fonts):
    """
    Zeichnet zwei Panels (Gestern | Heute) der PV-Leistung in Watt.
    """
    X0, X1 = 60, 800
    Y0, Y1 =   0, 160
    W, H    = X1-X0, Y1-Y0
    PW      = W/2

    vals_l = [e["power"] for e in left_data]
    vals_r = [e["power"] for e in right_data]
    vmin   = 0
    vmax   = max(vals_l + vals_r + [1])
    sy     = H/(vmax-vmin)

    # horizontale Linien + Y-Labels
    step = max(vmax//5, 1)
    yv = 0
    while yv <= vmax:
        y = Y1 - (yv-vmin)*sy
        d.line((X0, y, X1, y), fill=0)
        d.text((X0-45, y-7), f"{int(yv)}W", font=fonts["small"], fill=0)
        yv += step
    d.text((X0, Y0+5), "PV-Erzeugung (W)", font=fonts["small"], fill=0)

    # Gestern (linkes Panel)
    nL = len(vals_l)
    if nL > 1:
        xL = [X0 + i*(PW/(nL-1)) for i in range(nL)]
        for i in range(nL-1):
            d.line((xL[i],   Y1-(vals_l[i]  -vmin)*sy,
                    xL[i+1], Y1-(vals_l[i+1]-vmin)*sy),
                   fill=0, width=2)

    # Heute (rechtes Panel)
    nR = len(vals_r)
    if nR > 1:
        xR = [X0+PW + i*(PW/(nR-1)) for i in range(nR)]
        for i in range(nR-1):
            d.line((xR[i],   Y1-(vals_r[i]  -vmin)*sy,
                    xR[i+1], Y1-(vals_r[i+1]-vmin)*sy),
                   fill=0, width=2)

    # Untertitel
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    d.text((X0+10, Y1+5),       "Gestern", font=bf, fill=0)
    d.text((X0+PW+10, Y1+5),     "Heute",   font=bf, fill=0)

def main():
    # E-Paper initialisieren
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    # Leinwand
    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        "small": ImageFont.load_default(),
        "info":  ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    # PV-Daten abrufen
    left, right = get_historical_pv()

    # Chart zeichnen
    draw_pv_chart(d, left, right, fonts)

    # Update-Zeit unten links
    d.text((10, 470), time.strftime("Update: %H:%M %d.%m.%Y"), font=fonts["small"], fill=0)

    # Ausgabe & Sleep
    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()