
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime
import sqlite3
import logging

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# ————————————————————————————————————————————————————————————
# Logging
LOG_FILE = '/home/alex/logs/tibber.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Waveshare E-Paper Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key    # enthält API_KEY
DB_FILE = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'

# Cache-Dateien
CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'


def save_cache(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)


def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return None


def get_price_data():
    hdr = {
        "Authorization": f"Bearer {api_key.API_KEY}",
        "Content-Type": "application/json"
    }
    query = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    try:
        r = requests.post(
            "https://api.tibber.com/v1-beta/gql",
            json={"query": query},
            headers=hdr,
            timeout=20
        )
        r.raise_for_status()
        return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    except Exception as e:
        logging.error(f"Tibber-API-Fehler: {e}")
        return None


def update_price_cache(pi):
    today_str = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today_str:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today_str, "data": pi.get('today', [])}, CACHE_TODAY)


def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def prepare_data(pi):
    today_list = pi.get('today', [])
    vals_today = [s['total'] * 100 for s in today_list]
    lowest = min(vals_today) if vals_today else 0
    highest = max(vals_today) if vals_today else 0

    curr = pi.get('current', {})
    try:
        cur_dt = datetime.datetime.fromisoformat(curr['startsAt']).astimezone(local_tz)
        cur_price = curr['total'] * 100
    except Exception:
        cur_dt = datetime.datetime.now(local_tz)
        cur_price = 0

    slots = []
    for seg in pi.get('today', []) + pi.get('tomorrow', []):
        try:
            dt = datetime.datetime.fromisoformat(seg['startsAt']).astimezone(local_tz)
            slots.append((dt, seg['total'] * 100))
        except Exception:
            continue

    future = [(dt, p) for dt, p in slots if dt >= cur_dt]
    if future:
        ft, fv = min(future, key=lambda x: x[1])
        hours = round((ft - cur_dt).total_seconds() / 3600)
    else:
        hours, fv = 0, 0

    return {
        "current_price":    cur_price,
        "lowest_today":     lowest,
        "highest_today":    highest,
        "hours_to_lowest":  hours,
        "lowest_future_val": fv
    }


def get_pv_series(slots):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT ts, dtu_power FROM pv_log", conn)
    conn.close()

    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0)
    df.index = df.index.tz_localize(local_tz)

    last_ts = df.index.max()
    vals = []
    for s in slots:
        try:
            t = s['startsAt']
            dt = t if isinstance(t, datetime.datetime) else datetime.datetime.fromisoformat(t).astimezone(local_tz)
            if dt <= last_ts:
                v = df['dtu_power'].asof(dt)
                vals.append(float(v) if not pd.isna(v) else 0.0)
            else:
                vals.append(np.nan)
        except:
            vals.append(np.nan)
    return pd.Series(vals)


def draw_dashed_line(d, x1, y1, x2, y2, **kw):
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    dl, gl = kw.get('dash_length', 4), kw.get('gap_length', 4)
    step = dl + gl
    for i in range(int(dist/step) + 1):
        s = i * step
        e = min(s + dl, dist)
        rs, re = s/dist, e/dist
        xa, ya = x1 + dx*rs, y1 + dy*rs
        xb, yb = x1 + dx*re, y1 + dy*re
        d.line((xa, ya, xb, yb), fill=kw.get('fill', 0), width=kw.get('width', 1))


def draw_two_day_chart(d, left, right, fonts, subtitles, area, pv_y=None, pv_t=None, label_min_max=False):
    X0, Y0, X1, Y1 = area
    W, H = X1 - X0, Y1 - Y0
    PW = W / 2

    # ... (Panel-Zeichnung unverändert) ...

    # Y-Achse, Untertitel und draw_panel wie zuvor
    # siehe vorherige Implementation


def draw_info_box(d, info, fonts, y_start):
    # Infobox-Feld beginnt bei y_start
    X0, X1 = 60, epd7in5_V2.EPD().width
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    infos = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tagestief:   {info['lowest_today']/100:.2f}",
        f"Tageshoch:   {info['highest_today']/100:.2f}",
        f"Tiefst in {info['hours_to_lowest']}h"
    ]
    w = (X1 - X0) / len(infos)
    for i, t in enumerate(infos):
        d.text((X0 + i*w + 5, y_start), t, font=bf, fill=0)


def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    pi = get_price_data()
    if not pi:
        logging.error("Abbruch: keine Preisdaten.")
        return

    update_price_cache(pi)
    yesterday = get_cached_yesterday().get('data', [])
    today     = pi.get('today', [])
    tomorrow  = pi.get('tomorrow', [])

    # Chartbereich: 90% Breite, von 5% bis bottom-30px
    w, h = epd.width, epd.height
    mx = int(w * 0.05)
    my = int(h * 0.05)
    area = (mx, my, w - mx, h - 30)

    # Panel-Logik und Label-Modus
    if tomorrow:
        left_slots, right_slots = today, tomorrow
        pv_left, pv_right = get_pv_series(today), None
        subtitles = ("Preis & PV heute", "Preis & PV morgen")
        label_min_max = True
    else:
        left_slots, right_slots = yesterday, today
        pv_left, pv_right = get_pv_series(yesterday), get_pv_series(today)
        subtitles = ("Preise & PV gestern", "Preis & PV heute")
        label_min_max = False

    info = prepare_data(pi)

    img = Image.new('1', (w, h), 255)
    d = ImageDraw.Draw(img)
    fonts = {
        'small': ImageFont.load_default(),
        'info_font': ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    # Chart zeichnen
    draw_two_day_chart(
        d, left_slots, right_slots,
        fonts, subtitles, area,
        pv_y=pv_left, pv_t=pv_right,
        label_min_max=label_min_max
    )

    # Infobox direkt unter Chart (30px über bottom)
    chart_bottom_y = area[3]
    info_y = chart_bottom_y + 5
    draw_info_box(d, info, fonts, info_y)

    # Footer-Zeitstempel
    now_str = datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h - 20), now_str, font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Ungefangene Ausnahme im Hauptprogramm")

