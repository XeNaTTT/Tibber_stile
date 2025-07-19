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
DB_FILE         = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'
CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'

# Waveshare E-Paper Import
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# API-Key
import api_key

# Logging (to console for debugging)
logging.basicConfig(level=logging.DEBUG)

# --- Helper-Funktionen ---

def save_cache(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)

def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return None

def get_price_data():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}"}
    query = '''{ viewer { homes { currentSubscription { priceInfo { today { total startsAt } tomorrow { total startsAt } current { total startsAt } }}}}}'''
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": query}, headers=hdr)
    data = r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    logging.debug(f"Price data: {data}")
    return data

def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}

def prepare_data(pi):
    logging.debug("Preparing data...")
    today = [s['total']*100 for s in pi['today']]
    return {
        'current_price':   pi['current']['total']*100,
        'lowest_today':    min(today),
        'highest_today':   max(today),
        'hours_to_lowest': 0,
        'lowest_future_val': 0
    }

def get_pv_series(slots):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT ts,dtu_power FROM pv_log", conn)
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0)
    df.index = df.index.tz_localize(local_tz)

    pts = []
    for sl in slots:
        dt_obj = dt.datetime.fromisoformat(sl['startsAt']).astimezone(local_tz)
        pts.append(df['dtu_power'].asof(dt_obj))
    series = pd.Series(pts)
    logging.debug(f"PV series length {len(series)}: {series.tolist()}")
    return series

def draw_dashed_line(d, x1, y1, x2, y2, **kw):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    dl = kw.get('dash_length', 4)
    gl = kw.get('gap_length', 4)
    step = dl + gl
    for i in range(int(dist / step) + 1):
        s = i * step
        e = min(s + dl, dist)
        rs = s / dist
        re = e / dist
        xa = x1 + dx * rs
        ya = y1 + dy * rs
        xb = x1 + dx * re
        yb = y1 + dy * re
        d.line((xa, ya, xb, yb), fill=kw.get('fill', 0), width=kw.get('width', 1))

def draw_info_box(d, info, fonts, y):
    logging.debug(f"Drawing info box at y={y}")
    draw_list = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tief: {info['lowest_today']/100:.2f}",
        f"Hoch: {info['highest_today']/100:.2f}"
    ]
    for i, text in enumerate(draw_list):
        d.text((10 + i*120, y), text, font=fonts['small'], fill=0)

def draw_two_day_chart(d, left, right, fonts, subtitles, area, pv_y=None, pv_t=None, label_min_max=False):
    X0, Y0, X1, Y1 = area
    W, H = X1 - X0, Y1 - Y0
    PW = W / 2

    def extract(slots):
        times, prices = [], []
        for s in slots:
            try:
                dt_obj = dt.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
                times.append(dt_obj)
                prices.append(s['total'] * 100)
            except:
                continue
        return times, prices

    times_l, vals_l = extract(left)
    times_r, vals_r = extract(right)
    all_prices = vals_l + vals_r
    if not all_prices:
        return

    vmin, vmax = min(all_prices) - 0.5, max(all_prices) + 0.5
    sy = H / (vmax - vmin)

    pv_max = 0
    if pv_y is not None: pv_max = max(pv_max, pv_y.max())
    if pv_t is not None: pv_max = max(pv_max, pv_t.max())
    syv = H / (pv_max + 20) if pv_max > 0 else None

    # draw y-axis
    step = 5
    yv = math.floor(vmin / step) * step
    while yv <= vmax:
        yy = Y1 - (yv - vmin) * sy
        d.line((X0-5, yy, X0, yy), fill=0)
        d.line((X1, yy, X1+5, yy), fill=0)
        d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    d.text((X0-45, Y0-20), "Preis (ct/kWh)", font=fonts['small'], fill=0)

    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    d.text((X0+5, Y1+5),     subtitles[0], font=bf, fill=0)
    d.text((X0+PW+5, Y1+5),  subtitles[1], font=bf, fill=0)

    def panel(times, vals, pv_vals, x0):
        n = len(times)
        if n < 2:
            return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (vals[i]   - vmin)*sy
            x2, y2 = xs[i+1], Y1 - (vals[i+1] - vmin)*sy
            d.line((x1,y1,x2,y1), fill=0, width=2)
            d.line((x2,y1,x2,y2), fill=0, width=2)
        idx_min = vals.index(min(vals))
        idx_max = vals.index(max(vals))
        for idx in (idx_min, idx_max):
            xi, yi = xs[idx], Y1 - (vals[idx] - vmin)*sy
            d.text((xi-12, yi-12), f"{vals[idx]/100:.2f}", font=fonts['small'], fill=0)
        if syv and pv_vals is not None:
            pts = []
            for i in range(n):
                if not np.isnan(pv_vals.iloc[i]):
                    pts.append((xs[i], Y1 - int(pv_vals.iloc[i]*syv)))
                else:
                    pts.append(None)
            for a, b in zip(pts, pts[1:]):
                if a and b:
                    draw_dashed_line(d, a[0], a[1], b[0], b[1], dash_length=2, gap_length=2)
            valid = [i for i in range(n) if pts[i]]
            if valid:
                imax = max(valid, key=lambda i: pv_vals.iloc[i])
                xm, ym = pts[imax]
                d.text((xm-15, ym-15), f"{int(pv_vals.iloc[imax])}W", font=fonts['small'], fill=0)
        for i, t in enumerate(times):
            if i % 2 == 0:
                d.text((xs[i], Y1+18), t.strftime("%Hh"), font=fonts['small'], fill=0)

    panel(times_l, vals_l, pv_y,    X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(times_r, vals_r, pv_t, X0+PW)

# --- Hauptprogramm für Debug ---
if __name__ == '__main__':
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    pi = get_price_data()
    update_price_cache(pi)
    info = prepare_data(pi)

    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left, right = pi['today'], tomorrow
    else:
        left, right = get_cached_yesterday().get('data', []), pi['today']

    pv_left = get_pv_series(left)

    w, h = epd.width, epd.height
    mx = int(w * 0.05)
    img = Image.new('1', (w, h), 255)
    d = ImageDraw.Draw(img)
    fonts = {'small': ImageFont.load_default()}

    draw_info_box(d, info, fonts, 20)
    draw_two_day_chart(
        d, left, right, fonts,
        ("Gestern" if not tomorrow else "Heute", "Heute" if not tomorrow else "Morgen"),
        (mx, 40, w-mx, h-30),
        pv_y=pv_left
    )

    img.save('/home/alex/debug_epaper.png')
    print("Debug image saved to /home/alex/debug_epaper.png")
    sys.exit(0)
