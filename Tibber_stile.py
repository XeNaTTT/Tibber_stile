#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime as dt
import sqlite3
import logging

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Pfade
DB_FILE = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'
CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'

# Waveshare E-Paper Import
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# API-Key
import api_key

# Helper-Funktionen (unchanged...)
# ... save_cache, load_cache, get_price_data, update_price_cache, get_cached_yesterday,
# prepare_data, get_pv_series, draw_dashed_line, draw_info_box, draw_two_day_chart ...

# Hauptprogramm ohne stille Ausnahme-Abfangung
if __name__ == '__main__':
    # Initialisierung
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    # Hole Preisdaten
    pi = get_price_data()
    if not pi:
        print("ERROR: Keine Preisdaten (siehe Logging)" , file=sys.stderr)
        sys.exit(1)

    update_price_cache(pi)
    yest = get_cached_yesterday().get('data', [])
    td, tm = pi.get('today', []), pi.get('tomorrow', [])

    # Bildschirmabmessungen
    w, h = epd.width, epd.height
    mx = int(w * 0.05)

    # Erzeuge Bild und Draw-Objekt
    img = Image.new('1', (w, h), 255)
    d = ImageDraw.Draw(img)
    fonts = {
        'small': ImageFont.load_default(),
        'info_font': ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    # Zeichne Infobox oben
    info = prepare_data(pi)
    draw_info_box(d, info, fonts, y=20)

    # Definiere Chart-Bereich
    area = (mx, 40, w - mx, h - 30)

    # Wähle Daten für Link/Rechts-Panels
    if tm:
        left, right = td, tm
        pv_left, pv_right = get_pv_series(td), None
        subtitles = ("Preis & PV heute", "Preis & PV morgen")
        label_min_max = True
    else:
        left, right = yest, td
        pv_left, pv_right = get_pv_series(yest), get_pv_series(td)
        subtitles = ("Preise & PV gestern", "Preis & PV heute")
        label_min_max = False

    # Zeichne Chart
    draw_two_day_chart(
        d, left, right, fonts, subtitles, area,
        pv_y=pv_left, pv_t=pv_right,
        label_min_max=label_min_max
    )

    # Footer-Zeitstempel
    timestamp = dt.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h - 20), timestamp, font=fonts['small'], fill=0)

    # Anzeige auf E-Paper
    epd.display(epd.getbuffer(img))
    epd.sleep()

    # Optional: keine lange Pause, damit das Skript endet
    # sys.exit(0)
