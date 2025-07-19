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
    today = dt.date.today().isoformat(); ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}

def prepare_data(pi):
    logging.debug("Preparing data...")
    today = [s['total']*100 for s in pi['today']]
    return { 'current_price': pi['current']['total']*100, 'lowest_today': min(today), 'highest_today': max(today), 'hours_to_lowest':0, 'lowest_future_val':0 }

def get_pv_series(slots):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT ts,dtu_power FROM pv_log", conn)
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'], unit='s'); df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0); df.index = df.index.tz_localize(local_tz)
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

# --- next function ---
def draw_info_box(d, info, fonts, y):
    logging.debug(f"Drawing info box at y={y}")
    draw_list = [f"Preis jetzt: {info['current_price']/100:.2f}", f"Tief: {info['lowest_today']/100:.2f}", f"Hoch: {info['highest_today']/100:.2f}"]
    for i,text in enumerate(draw_list): d.text((10+i*100,y), text, font=fonts['small'], fill=0)

def draw_two_day_chart(d,left,right,fonts,subs,area,pv_y=None,pv_t=None,label_min_max=False):
    logging.debug(f"Chart area: {area}, left slots: {len(left)}, right slots: {len(right)}")
    X0,Y0,X1,Y1 = area; W,H=X1-X0,Y1-Y0
    # draw border for debugging
    d.rectangle(area, outline=0)
    # dummy line
    d.line((X0,Y1,X1,Y0),fill=0)

# --- Hauptprogramm für Debug ---
if __name__ == '__main__':
    # Initialize EPD (for dimensions)
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    # Fetch and prepare data
    pi = get_price_data()
    update_price_cache(pi)
    info = prepare_data(pi)

    # Determine panels based on tomorrow data
    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left = pi['today']
        right = tomorrow
    else:
        left = get_cached_yesterday().get('data', [])
        right = pi['today']

    # Get PV series for left (yesterday or today)
    pv_left = get_pv_series(left)

    # Canvas setup
    w, h = epd.width, epd.height
    mx = int(w * 0.05)
    img = Image.new('1', (w, h), 255)
    d = ImageDraw.Draw(img)
    fonts = {'small': ImageFont.load_default()}

    # Draw debug info box and chart
    draw_info_box(d, info, fonts, 20)
    draw_two_day_chart(
        d, left, right, fonts,
        ("Links","Rechts"),
        (mx, 40, w-mx, h-30),
        pv_y=pv_left
    )

    # Save debug image to filesystem rather than EPD
    debug_path = '/home/alex/debug_epaper.png'
    img.save(debug_path)
    print(f'Debug image saved to {debug_path}')
    sys.exit(0)
