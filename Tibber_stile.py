#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime
from PIL import Image, ImageDraw, ImageFont

# Zeitzone (Python 3.9+ oder backports)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

CACHE_FILE_TODAY     = '/home/alex/cached_today_price.json'
CACHE_FILE_YESTERDAY = '/home/alex/cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)

def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return None

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_FILE_TODAY)
    if not ct or ct.get('date') != today:
        if ct:
            save_cache(ct, CACHE_FILE_YESTERDAY)
        save_cache({"date": today, "data": pd['today']}, CACHE_FILE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_FILE_YESTERDAY) or {"data": []}

def get_price_data():
    hdr = {"Authorization": "Bearer " + api_key.API_KEY, "Content-Type": "application/json"}
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query": q}, headers=hdr, timeout=15)
    r.raise_for_status()
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def prepare_data(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur_iso = pd['current']['startsAt']
    cur_dt = datetime.datetime.fromisoformat(cur_iso).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    return {
        "current_price": cur_price,
        "current_dt":    cur_dt,
        "lowest_today":  lowest_today,
        "highest_today": highest_today
    }

def draw_dashed_line(d, x1,y1,x2,y2, **kw):
    dx,dy = x2-x1, y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get("dash_length",4), kw.get("gap_length",4)
    step = dl+gl
    for i in range(int(dist/step)+1):
        s = i*step
        e = min(s+dl,dist)
        rs, re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get("fill",0), width=kw.get("width",1))

def draw_two_day_chart(d, left_data, right_data, fonts, mode):
    # Chart-Rahmen
    X0,X1 = 60, 800
    Y0,Y1 = 50, 400
    W, H  = X1-X0, Y1-Y0
    PW    = W/2

    # Werte extrahieren
    def extract(data):
        ts, vals = [], []
        for s in data:
            dt_obj = datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
            ts.append(dt_obj)
            vals.append(s['total']*100)
        return ts, vals

    tl, vl = extract(left_data)
    tr, vr = extract(right_data)
    allv = vl + vr
    if allv:
        vmin, vmax = min(allv)-0.5, max(allv)+0.5
    else:
        vmin, vmax = 0,1
    sy = H/(vmax-vmin)

    # PV-Skalierung (nur um H-Bereich anzupassen)
    # PV wird nur als Overlay gezeichnet wenn vorhanden
    # falls nicht vorhanden: übersprungen

    # Y-Achse
    step = 5
    yv = math.floor(vmin/step)*step
    while yv <= vmax:
        y = Y1 - (yv-vmin)*sy
        d.line((X0-5, y, X0, y), fill=0)
        d.line((X1, y, X1+5, y), fill=0)
        d.text((X0-45, y-7), f"{yv/100:.2f}", font=fonts["small"], fill=0)
        yv += step
    d.text((X0-45, Y0-20), "Preis (ct/kWh)", font=fonts["small"], fill=0)

    # Linkes Panel
    nL = len(tl)
    xL = [X0 + i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]
    for i in range(nL-1):
        x1,y1 = xL[i],   Y1 - (vl[i]   - vmin)*sy
        x2,y2 = xL[i+1], Y1 - (vl[i+1] - vmin)*sy
        d.line((x1,y1,x2,y1), fill=0, width=2)
        d.line((x2,y1,x2,y2), fill=0, width=2)
    for i, dt in enumerate(tl):
        if i%2==0:
            d.text((xL[i], Y1+5), dt.strftime('%Hh'), font=fonts["small"], fill=0)

    # Mittellinie
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

    # Rechtes Panel
    nR = len(tr)
    xR = [X0+PW + i*(PW/(nR-1)) for i in range(nR)] if nR>1 else [X0+PW]
    for i in range(nR-1):
        x1,y1 = xR[i],   Y1 - (vr[i]   - vmin)*sy
        x2,y2 = xR[i+1], Y1 - (vr[i+1] - vmin)*sy
        d.line((x1,y1,x2,y1), fill=0, width=2)
        d.line((x2,y1,x2,y2), fill=0, width=2)
    for i, dt in enumerate(tr):
        if i%2==0:
            d.text((xR[i], Y1+5), dt.strftime('%Hh'), font=fonts["small"], fill=0)

    # Marker: gefüllter Kreis an aktuellem Punkt
    now = datetime.datetime.now(local_tz)
    # bei historischem Modus im rechten Panel, sonst im linken
    if mode == 'historical' or not right_data:
        ts_used, vs_used, xs_used = tr, vr, xR
    else:
        ts_used, vs_used, xs_used = tl, vl, xL

    for i in range(len(ts_used)-1):
        if ts_used[i] <= now < ts_used[i+1]:
            frac = (now - ts_used[i]).total_seconds() / (ts_used[i+1] - ts_used[i]).total_seconds()
            px = xs_used[i] + frac * (xs_used[i+1] - xs_used[i])
            py = Y1 - (vs_used[i] - vmin)*sy
            r = 4
            d.ellipse((px-r, py-r, px+r, py+r), fill=0)
            break

def draw_info_box(d, data, fonts):
    X0,X1 = 60,800
    y     = 420
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    texts = [
        f"Aktuell: {data['current_price']/100:.2f}",
        f"Tief:    {data['lowest_today']/100:.2f}",
        f"Hoch:    {data['highest_today']/100:.2f}"
    ]
    w = (X1 - X0) / len(texts)
    for i,t in enumerate(texts):
        d.text((X0 + i*w + 5, y), t, font=bf, fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday()
    info = prepare_data(pd)

    if pd['tomorrow']:
        mode = 'future'
        left_data  = pd['today']
        right_data = pd['tomorrow']
    else:
        mode = 'historical'
        left_data  = cy.get('data', [])
        right_data = pd['today']

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {"small": ImageFont.load_default()}

    draw_two_day_chart(d, left_data, right_data, fonts, mode)
    draw_info_box(d, info, fonts)

    d.text((10, epd.height-20),
           time.strftime("Update: %H:%M %d.%m.%Y"),
           font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)

if __name__=="__main__":
    main()