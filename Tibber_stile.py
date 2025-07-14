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

# Waveshare E-Paper Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key  # enthält API_KEY
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
        payload = r.json()
        return payload['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    except Exception as e:
        logging.error(f"Tibber-API-Fehler: {e}")
        return None


def update_price_cache(price_info):
    today_str = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today_str:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today_str, "data": price_info.get('today', [])}, CACHE_TODAY)


def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def prepare_data(price_info):
    today_list = price_info.get('today', [])
    vals_today = [s['total'] * 100 for s in today_list]
    lowest = min(vals_today) if vals_today else 0
    highest = max(vals_today) if vals_today else 0

    curr = price_info.get('current', {})
    try:
        cur_dt = datetime.datetime.fromisoformat(curr['startsAt']).astimezone(local_tz)
        cur_price = curr['total'] * 100
    except Exception:
        cur_dt = datetime.datetime.now(local_tz)
        cur_price = 0

    slots = []
    for seg in price_info.get('today', []) + price_info.get('tomorrow', []):
        try:
            dt = datetime.datetime.fromisoformat(seg['startsAt']).astimezone(local_tz)
            slots.append((dt, seg['total'] * 100))
        except Exception:
            continue

    future = [(dt, price) for dt, price in slots if dt >= cur_dt]
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

    pv_vals = []
    for s in slots:
        try:
            t = s['startsAt']
            dt = (t if isinstance(t, datetime.datetime)
                  else datetime.datetime.fromisoformat(t).astimezone(local_tz))
            v = df['dtu_power'].asof(dt)
            pv_vals.append(float(v) if not pd.isna(v) else 0.0)
        except Exception:
            pv_vals.append(0.0)
    return pd.Series(pv_vals)


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
        rs, re = s / dist, e / dist
        xa, ya = x1 + dx * rs, y1 + dy * rs
        xb, yb = x1 + dx * re, y1 + dy * re
        d.line((xa, ya, xb, yb), fill=kw.get('fill', 0), width=kw.get('width', 1))


def draw_two_day_chart(d, left, right, fonts, subtitles, area, pv_y=None, pv_t=None):
    X0, Y0, X1, Y1 = area
    pad = 10
    X0 += pad; Y0 += pad; X1 -= pad; Y1 -= pad
    W = X1 - X0; H = Y1 - Y0; PW = W / 2

    # extrahiere Zeiten & Preise
    def extract(slots):
        times, prices = [], []
        for slot in slots:
            if isinstance(slot, dict):
                t = slot.get('startsAt')
                p = slot.get('total', 0) * 100
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

    vmin, vmax = min(allp) - 0.5, max(allp) + 0.5
    sy_p = H / (vmax - vmin)

    # PV-Skalierung
    sy_v = None
    if pv_y is not None and pv_t is not None:
        pm = max(pv_y.max(), pv_t.max(), 0)
        if pm > 0:
            sy_v = H / (pm + 20)

    # Y-Achse
    step = 5
    y0 = math.floor(vmin/step) * step
    while y0 <= vmax:
        y = Y1 - (y0 - vmin) * sy_p
        d.line((X0-5, y, X0, y), fill=0)
        d.line((X1, y, X1+5, y), fill=0)
        d.text((X0-45, y-7), f"{y0/100:.2f}", font=fonts['small'], fill=0)
        y0 += step
    d.text((X0-45, Y0-20), 'Preis (ct/kWh)', font=fonts['small'], fill=0)

    # Untertitel
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    d.text((X0+10, Y1+10),     subtitles[0], font=bf, fill=0)
    d.text((X0+PW+10, Y1+10),  subtitles[1], font=bf, fill=0)

    # Panel-Zeichner
    def draw_panel(times, vals, pv_vals, x0):
        n = len(times)
        if n < 2:
            return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        # Preislinie
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (vals[i]   - vmin)*sy_p
            x2, y2 = xs[i+1], Y1 - (vals[i+1] - vmin)*sy_p
            d.line((x1,y1,x2,y1), fill=0, width=2)
            d.line((x2,y1,x2,y2), fill=0, width=2)
            mx = (x1 + x2)/2
            d.text((mx-12, y1-12), f"{vals[i]/100:.2f}", font=fonts['small'], fill=0)
        # PV-Overlay (falls passend)
        if sy_v and pv_vals is not None and len(pv_vals) == n:
            pts = [(xs[i], Y1 - int(pv_vals.iloc[i]*sy_v)) for i in range(n)]
            for a, b in zip(pts, pts[1:]):
                draw_dashed_line(d, a[0], a[1], b[0], b[1], dash_length=2, gap_length=2)
            imax = int(pv_vals.idxmax())
            xm, ym = pts[imax]
            d.text((xm-15, ym-15), f"{int(pv_vals.iloc[imax])}W", font=fonts['small'], fill=0)
        # Zeitachse
        for i, dt in enumerate(times):
            if i % 2 == 0:
                d.text((xs[i], Y1+25), dt.strftime('%Hh'), font=fonts['small'], fill=0)

    # links
    draw_panel(times_l, vals_l, pv_y, X0)
    # Mittellinie
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    # rechts
    draw_panel(times_r, vals_r, pv_t, X0+PW)


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

    price_info = get_price_data()
    if not price_info:
        logging.error("Abbruch: keine Preisdaten.")
        return

    update_price_cache(price_info)
    yesterday = get_cached_yesterday().get('data', [])
    today     = price_info.get('today', [])
    tomorrow  = price_info.get('tomorrow', [])

    if tomorrow:
        # links=heute+PV, rechts=morgen+keine PV
        left_slots,  right_slots = today, tomorrow
        pv_left,     pv_right   = get_pv_series(today), None
        subtitles = ("Preis & PV heute", "Preis & PV morgen")
    else:
        # links=gestern+PV, rechts=heute+PV
        left_slots,  right_slots = yesterday, today
        pv_left,     pv_right   = get_pv_series(yesterday), get_pv_series(today)
        subtitles = ("Preise & PV gestern", "Preis & PV heute")

    info = prepare_data(price_info)

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        'small':     ImageFont.load_default(),
        'info_font': ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    upper = (0, 0, epd.width, int(epd.height * 0.6))
    draw_two_day_chart(d, left_slots, right_slots, fonts, subtitles, upper, pv_left, pv_right)
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
