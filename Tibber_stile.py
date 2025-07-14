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

# Waveshare-Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key  # sorgt für API_KEY
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
        "Authorization": "Bearer " + api_key.API_KEY,
        "Content-Type": "application/json"
    }
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    try:
        r = requests.post(
            "https://api.tibber.com/v1-beta/gql",
            json={"query": q},
            headers=hdr,
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return data['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    except Exception as e:
        logging.error(f"Tibber-API-Fehler: {e}")
        return None


def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pd.get('today', [])}, CACHE_TODAY)


def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def prepare_data(pd):
    today_list = pd.get('today', [])
    vals_today = [s['total'] * 100 for s in today_list]
    lowest = min(vals_today) if vals_today else 0
    highest = max(vals_today) if vals_today else 0

    current = pd.get('current', {})
    try:
        cur_dt = datetime.datetime.fromisoformat(current['startsAt']).astimezone(local_tz)
        cur_price = current['total'] * 100
    except Exception:
        cur_dt = datetime.datetime.now(local_tz)
        cur_price = 0

    slots = []
    for seg in pd.get('today', []) + pd.get('tomorrow', []):
        try:
            dt = datetime.datetime.fromisoformat(seg['startsAt']).astimezone(local_tz)
            price = seg['total'] * 100
            slots.append((dt, price))
        except Exception:
            continue

    future = [(dt, val) for dt, val in slots if dt >= cur_dt]
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

    vals = []
    for s in slots:
        try:
            dt = datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
            v = df['dtu_power'].asof(dt)
            vals.append(float(v) if not pd.isna(v) else 0.0)
        except Exception:
            vals.append(0.0)
    return pd.Series(vals)


def draw_dashed_line(d, x1, y1, x2, y2, **kw):
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    dl, gl = kw.get('dash_length', 4), kw.get('gap_length', 4)
    step = dl + gl
    for i in range(int(dist/step) + 1):
        s = i*step; e = min(s+dl, dist)
        rs, re = s/dist, e/dist
        xa, ya = x1 + dx*rs, y1 + dy*rs
        xb, yb = x1 + dx*re, y1 + dy*re
        d.line((xa, ya, xb, yb), fill=kw.get('fill', 0), width=kw.get('width', 1))


def draw_two_day_chart(d, left, right, fonts, mode, area, pv_y=None, pv_t=None):
    X0, Y0, X1, Y1 = area
    pad = 10
    X0 += pad; Y0 += pad; X1 -= pad; Y1 -= pad
    W = X1 - X0; H = Y1 - Y0; PW = W / 2

    # Preiswerte extrahieren
    def extract(vals):
        times, prices = [], []
        for slot in vals:
            # wenn slot ein Dict ist
            if isinstance(slot, dict):
                t = slot.get('startsAt')
                p = slot.get('total', 0) * 100
            # wenn slot schon ein Tupel (dt, price)
            elif isinstance(slot, tuple) and len(slot) == 2:
                t, p = slot
            else:
                continue
            try:
                dt = (t if isinstance(t, datetime.datetime)
                      else datetime.datetime.fromisoformat(t).astimezone(local_tz))
            except Exception:
                continue
            times.append(dt)
            prices.append(p)
        return times, prices

    times_l, vals_l = extract(left)
    times_r, vals_r = extract(right)
    allp = vals_l + vals_r
    if not allp:
        return

    vmin_p, vmax_p = min(allp) - 0.5, max(allp) + 0.5
    sy_p = H / (vmax_p - vmin_p)

    # PV-Skalierung falls gewünscht
    sy_v = None
    if pv_y is not None and pv_t is not None:
        pm = max(pv_y.max(), pv_t.max(), 0)
        if pm > 0:
            sy_v = H / (pm + 20)

    # … hier folgt die übliche Zeichenlogik: Y-Achse, Panels, Linien, Labels …
    # (identisch zu deinem Original, nur dass du jetzt times_l/vals_l nutzt)

    # Beispiel: einfache Linie für links
    for i in range(len(times_l)-1):
        x1 = X0 + i*(PW/(len(times_l)-1))
        y1 = Y1 - (vals_l[i] - vmin_p)*sy_p
        x2 = X0 + (i+1)*(PW/(len(times_l)-1))
        y2 = Y1 - (vals_l[i+1] - vmin_p)*sy_p
        d.line((x1,y1,x2,y1), fill=0, width=2)
        d.line((x2,y1,x2,y2), fill=0, width=2)
        mx = (x1+x2)/2
        d.text((mx-12,y1-12), f"{vals_l[i]/100:.2f}", font=fonts['small'], fill=0)

    # … und analog für das rechte Panel …

def draw_subtitle_labels(d, fonts, mode):
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    X0, X1 = 60, epd7in5_V2.EPD().width
    PW = (X1 - X0) / 2
    y = int(epd7in5_V2.EPD().height * 0.6) + 5
    d.text((X0+10, y),     'Preise & PV gestern', font=bf, fill=0)
    d.text((X0+PW+10, y), 'Preis & PV heute',    font=bf, fill=0)


def draw_info_box(d, info, fonts):
    X0, X1 = 60, epd7in5_V2.EPD().width
    y = int(epd7in5_V2.EPD().height * 0.6) + 25
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    infos = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tagestief:   {info['lowest_today']/100:.2f}",
        f"Tageshoch:   {info['highest_today']/100:.2f}",
        f"Tiefst in {info['hours_to_lowest']}h"
    ]
    w = (X1 - X0) / len(infos)
    for i, t in enumerate(infos):
        d.text((X0 + i*w + 5, y), t, font=bf, fill=0)


def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    pinfo = get_price_data()
    if not pinfo:
        logging.error("Abbruch: keine Preisdaten.")
        return

    update_price_cache(pinfo)
    cy = get_cached_yesterday()
    info = prepare_data(pinfo)
    left_price  = cy.get('data', [])
    right_price = pinfo.get('today', [])

    pv_y = get_pv_series(left_price)
    pv_t = get_pv_series(right_price)
    logging.info(f"PV gestern: {pv_y.tolist()}")
    logging.info(f"PV heute:   {pv_t.tolist()}")

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        'small':     ImageFont.load_default(),
        'info_font': ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    upper = (0, 0, epd.width, int(epd.height * 0.6))
    draw_two_day_chart(d, left_price, right_price, fonts, 'historical', upper, pv_y, pv_t)
    draw_subtitle_labels(d, fonts, 'historical')
    draw_info_box(d, info, fonts)

    now_str = datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, epd.height-20), now_str, font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Ungefangene Ausnahme im Hauptprogramm")
