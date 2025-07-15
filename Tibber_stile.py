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

# E-Paper Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key    # enthält API_KEY
DB_FILE = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'

CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'

# ---------------- Helper-Funktionen ----------------

def save_cache(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)


def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return None


def get_price_data():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    query = '''
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    '''
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
        logging.error(f"API-Fehler: {e}")
        return None


def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi.get('today', [])}, CACHE_TODAY)


def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def prepare_data(pi):
    today_list = pi.get('today', [])
    vals = [s['total']*100 for s in today_list]
    lowest = min(vals) if vals else 0
    highest = max(vals) if vals else 0
    cur = pi.get('current', {})
    try:
        cur_dt = dt.datetime.fromisoformat(cur['startsAt']).astimezone(local_tz)
        cur_price = cur['total']*100
    except:
        cur_dt = dt.datetime.now(local_tz)
        cur_price = 0
    slots = []
    for seg in pi.get('today', []) + pi.get('tomorrow', []):
        try:
            dt_obj = dt.datetime.fromisoformat(seg['startsAt']).astimezone(local_tz)
            slots.append((dt_obj, seg['total']*100))
        except:
            pass
    future = [(d,p) for d,p in slots if d >= cur_dt]
    if future:
        ft, fv = min(future, key=lambda x: x[1])
        hours = round((ft - cur_dt).total_seconds() / 3600)
    else:
        hours, fv = 0, 0
    return {
        "current_price": cur_price,
        "lowest_today": lowest,
        "highest_today": highest,
        "hours_to_lowest": hours,
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
    for ts, _ in slots:
        if ts <= last_ts:
            v = df['dtu_power'].asof(ts)
            vals.append(float(v) if not pd.isna(v) else 0.0)
        else:
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


def draw_info_box(d, info, fonts, y):
    X0, X1 = 60, epd7in5_V2.EPD().width
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    texts = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tagestief:   {info['lowest_today']/100:.2f}",
        f"Tageshoch:   {info['highest_today']/100:.2f}",
        f"Tiefst in {info['hours_to_lowest']}h"
    ]
    w = (X1 - X0) / len(texts)
    for i, t in enumerate(texts):
        d.text((X0 + i*w + 5, y), t, font=bf, fill=0)


def draw_two_day_chart(d, left, right, fonts, subtitles, area, pv_y=None, pv_t=None, label_min_max=False):
    X0, Y0, X1, Y1 = area
    W, H = X1 - X0, Y1 - Y0
    PW = W / 2

    # extract times & prices
    def extract(slots):
        times, prices = [], []
        for t, p in slots:
            times.append(t)
            prices.append(p)
        return times, prices

    times_l, vals_l = extract(left)
    times_r, vals_r = extract(right)
    allvals = vals_l + vals_r
    if not allvals:
        return

    vmin, vmax = min(allvals) - 0.5, max(allvals) + 0.5
    sy = H / (vmax - vmin)

    # PV scale
    pv_max = max(pv_y.max() if pv_y is not None else 0,
                 pv_t.max() if pv_t is not None else 0)
    syv = H / (pv_max + 20) if pv_max > 0 else None

    # y-axis
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        yy = Y1 - (yv - vmin) * sy
        d.line((X0-5, yy, X0, yy), fill=0)
        d.line((X1, yy, X1+5, yy), fill=0)
        d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    d.text((X0-45, Y0-20), 'Preis (ct/kWh)', font=fonts['small'], fill=0)

    # subtitles
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    d.text((X0+5, Y1+5),     subtitles[0], font=bf, fill=0)
    d.text((X0+PW+5, Y1+5),  subtitles[1], font=bf, fill=0)

    def panel(times, vals, pv_vals, x0):
        n = len(times)
        if n<2: return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        # price line
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (vals[i]   - vmin)*sy
            x2, y2 = xs[i+1], Y1 - (vals[i+1] - vmin)*sy
            d.line((x1,y1,x2,y1), fill=0, width=2)
            d.line((x2,y1,x2,y2), fill=0, width=2)
        # label min/max
        idx_min = vals.index(min(vals))
        idx_max = vals.index(max(vals))
        for idx in (idx_min, idx_max):
            xi = xs[idx]
            yi = Y1 - (vals[idx] - vmin)*sy
            d.text((xi-12, yi-12), f"{vals[idx]/100:.2f}", font=fonts['small'], fill=0)
        # pv overlay
        if syv and pv_vals is not None:
            pts = [(xs[i], Y1 - int(pv_vals.iloc[i]*syv)) if not np.isnan(pv_vals.iloc[i]) else None
                   for i in range(n)]
            for a,b in zip(pts, pts[1:]):
                if a and b:
                    draw_dashed_line(d, a[0],a[1], b[0],b[1], dash_length=2, gap_length=2)
            valid = [i for i,p in enumerate(pv_vals) if not np.isnan(p)]
            if valid:
                imax = max(valid, key=lambda i: pv_vals.iloc[i])
                xm, ym = pts[imax]
                d.text((xm-15, ym-15), f"{int(pv_vals.iloc[imax])}W", font=fonts['small'], fill=0)
        # time axis
        for i, dt in enumerate(times):
            if i%2==0:
                d.text((xs[i], Y1+18), dt.strftime('%Hh'), font=fonts['small'], fill=0)

    # draw both panels
    panel(times_l, vals_l, pv_y,    X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(times_r, vals_r, pv_t, X0+PW)

# ==================================
# Hauptprogramm
# ==================================

def main():
    epd = epd7in5_V2.EPD(); epd.init(); epd.Clear()
    pi = get_price_data()
    if not pi:
        logging.error("Keine Preisdaten")
        return
    update_price_cache(pi)
    yest = get_cached_yesterday().get('data', [])
    td, tm = pi.get('today', []), pi.get('tomorrow', [])
    w, h = epd.width, epd.height
    mx = int(w * 0.05)

    # Infobox oben
    img = Image.new('1', (w, h), 255)
    d = ImageDraw.Draw(img)
    fonts = {
        'small': ImageFont.load_default(),
        'info_font': ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }
    info = prepare_data(pi)
    draw_info_box(d, info, fonts, 20)

    # Chartbereich
    area = (mx, 40, w - mx, h - 30)
    if tm:
        ls, rs = td, tm
        pv_l, pv_r = get_pv_series(td), None
        subs = ("Preis & PV heute", "Preis & PV morgen")
        lm = True
    else:
        ls, rs = yest, td
        pv_l, pv_r = get_pv_series(yest), get_pv_series(td)
        subs = ("Preise & PV gestern", "Preis & PV heute")
        lm = False

    draw_two_day_chart(
        d, ls, rs, fonts, subs, area,
        pv_y=pv_l, pv_t=pv_r, label_min_max=lm
    )

    # Footer
    timestamp = dt.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h - 20), timestamp, font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fehler im Hauptprogramm")
