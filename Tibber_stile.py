#!/usr/bin/env python3
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

# Waveshare‑Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# API‑Key
import api_key

logging.basicConfig(level=logging.INFO)

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
    query = '''
    { viewer { homes { currentSubscription { priceInfo {
        today    { total startsAt }
        tomorrow { total startsAt }
        current  { total startsAt }
    }}}}}
    '''
    r = requests.post(
        "https://api.tibber.com/v1-beta/gql",
        json={"query": query},
        headers=hdr,
        timeout=15
    )
    r.raise_for_status()
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

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
    today_vals = [s['total']*100 for s in pi['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur = pi['current']
    cur_dt    = dt.datetime.fromisoformat(cur['startsAt']).astimezone(local_tz)
    cur_price = cur['total']*100
    return {
        'current_dt':    cur_dt,
        'current_price': cur_price,
        'lowest_today':  lowest_today,
        'highest_today': highest_today
    }

def get_pv_series(slots):
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql_query("SELECT ts, dtu_power FROM pv_log", conn)
    conn.close()

    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0)
    df.index = df.index.tz_localize(local_tz)

    last_ts = df.index.max()
    vals    = []
    for s in slots:
        # String → dt.datetime
        dt_obj = dt.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
        if dt_obj <= last_ts:
            v = df['dtu_power'].asof(dt_obj)
            vals.append(float(v) if not pd.isna(v) else 0.0)
        else:
            # künftige Zeiten als 0
            vals.append(0.0)
    return pd.Series(vals)

def draw_dashed_line(d, x1, y1, x2, y2, **kw):
    dx,dy = x2 - x1, y2 - y1
    dist  = math.hypot(dx, dy)
    if dist == 0:
        return
    dl = kw.get('dash_length', 4)
    gl = kw.get('gap_length', 4)
    step = dl + gl
    for i in range(int(dist/step) + 1):
        s = i * step
        e = min(s + dl, dist)
        rs, re = s/dist, e/dist
        xa = x1 + dx*rs
        ya = y1 + dy*rs
        xb = x1 + dx*re
        yb = y1 + dy*re
        d.line((xa, ya, xb, yb), fill=kw.get('fill', 0), width=1)

def draw_info_box(d, info, fonts, y):
    X0, X1 = 60, epd7in5_V2.EPD().width
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    texts = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tief heute:  {info['lowest_today']/100:.2f}",
        f"Hoch heute:  {info['highest_today']/100:.2f}"
    ]
    w = (X1 - X0) / len(texts)
    for i, t in enumerate(texts):
        d.text((X0 + i*w + 5, y), t, font=bf, fill=0)

def draw_two_day_chart(d, left, right, fonts, subtitles, area,
                       pv_y=None, pv_t=None, cur_dt=None, cur_price=None):
    X0, Y0, X1, Y1 = area
    W, H = X1 - X0, Y1 - Y0
    PW    = W / 2

    def extract(slots):
        ts, ps = [], []
        for s in slots:
            dt_obj = dt.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
            ts.append(dt_obj)
            ps.append(s['total']*100)
        return ts, ps

    tl, vl = extract(left)
    tr, vr = extract(right)
    allp   = vl + vr
    if not allp:
        return

    vmin, vmax = min(allp) - 0.5, max(allp) + 0.5
    sy   = H / (vmax - vmin)
    pv_max = 0
    if pv_y is not None: pv_max = max(pv_max, pv_y.max())
    if pv_t is not None: pv_max = max(pv_max, pv_t.max())
    syv = H / (pv_max + 20) if pv_max > 0 else None

    # Y‑Achse
    step = 5
    yv   = math.floor(vmin/step)*step
    while yv <= vmax:
        yy = Y1 - (yv - vmin)*sy
        d.line((X0-5, yy, X0, yy), fill=0)
        d.line((X1, yy, X1+5, yy), fill=0)
        d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    d.text((X0-45, Y0-20), 'Preis (ct/kWh)', font=fonts['small'], fill=0)

    # Überschriften unter dem Chart
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    d.text((X0+5, Y1+5),    subtitles[0], font=bf, fill=0)
    d.text((X0+PW+5, Y1+5), subtitles[1], font=bf, fill=0)

    def panel(ts_list, val_list, pv_list, x0):
        n = len(ts_list)
        if n < 2:
            return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        # Preis-Polyline
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (val_list[i]   - vmin)*sy
            x2, y2 = xs[i+1], Y1 - (val_list[i+1] - vmin)*sy
            d.line((x1,y1, x2,y1), fill=0, width=2)
            d.line((x2,y1, x2,y2), fill=0, width=2)
        # Tiefst- und Höchstpreis labeln
        for idx in (val_list.index(min(val_list)), val_list.index(max(val_list))):
            xi, yi = xs[idx], Y1 - (val_list[idx] - vmin)*sy
            d.text((xi-12, yi-12), f"{val_list[idx]/100:.2f}", font=fonts['small'], fill=0)
        # PV‑Overlay (zukünftige Werte auf 0)
        if syv and pv_list is not None:
            pts = []
            for i in range(n):
                v = pv_list.iloc[i] if not np.isnan(pv_list.iloc[i]) else 0.0
                pts.append((xs[i], Y1 - int(v*syv)))
            for a, b in zip(pts, pts[1:]):
                draw_dashed_line(d, a[0], a[1], b[0], b[1], dash_length=2, gap_length=2)

    panel(tl, vl, pv_y, X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(tr, vr, pv_t, X0+PW)

    # Marker: gefüllter Kreis auf aktuellem Preis plus Label
    if cur_dt and cur_price is not None:
        # Index der aktuellen Stunde im rechten Panel suchen
        idx = next((i for i, t in enumerate(tr) if t.hour == cur_dt.hour), None)
        if idx is not None and len(tr) > 1:
            # x‑Position berechnen
            px = X0 + PW + idx * (PW / (len(tr) - 1))
            # y‑Position exakt auf der Preis‑Kurve
            py = Y1 - (cur_price - vmin) * sy
            r = 4
            # gefüllter Kreis
            d.ellipse((px - r, py - r, px + r, py + r), fill=0)
            # Label mit aktuellem Preis (ct/kWh)
            price_txt = f"{cur_price / 100:.2f}"
            d.text((px + r + 2, py - r - 2), price_txt, font=fonts['small'], fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    pi   = get_price_data()
    update_price_cache(pi)
    info = prepare_data(pi)

    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left, right = pi['today'], tomorrow
        labels      = ("Heute", "Morgen")
    else:
        left, right = get_cached_yesterday()['data'], pi['today']
        labels      = ("Gestern", "Heute")

    pv_left  = get_pv_series(left)
    pv_right = get_pv_series(right)

    w, h = epd.width, epd.height
    mx   = int(w * 0.05)
    img  = Image.new('1', (w, h), 255)
    d    = ImageDraw.Draw(img)
    fonts = {'small': ImageFont.load_default()}

    draw_info_box(d, info, fonts, 20)
    draw_two_day_chart(
        d, left, right, fonts, labels, (mx, 40, w-mx, h-30),
        pv_y= pv_left,
        pv_t= pv_right,
        cur_dt= info['current_dt'],
        cur_price= info['current_price']
    )

    # Footer mit Zeitstempel
    ts = dt.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h-20), ts, font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()
