#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import time
import math
import json
import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
from zoneinfo import ZoneInfo

# Lokale Zeitzone
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare‐Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

# DTU‐SN und lokale Web‐API (Docker im Host‐Netzwerk)
DTU_SN   = "DTUBI-4143A019CB05"
API_BASE = "http://127.0.0.1:5000"

# Tibber‐API‐Key
import api_key

# Cache‐Dateien für Strompreise
CACHE_TODAY_PRICE     = 'cached_today_price.json'
CACHE_YESTERDAY_PRICE = 'cached_yesterday_price.json'

# ——— Hilfsfunktionen für Caching ———
def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data, f)

def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

# ——— Strompreis‐Datenerfassung ———
def get_price_data():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    query = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": query}, headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY_PRICE)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY_PRICE)
        save_cache({"date": today, "data": pd['today']}, CACHE_TODAY_PRICE)

def get_cached_yesterday_price():
    cy = load_cache(CACHE_YESTERDAY_PRICE)
    return cy.get('data', []) if cy else []

def get_consumption_data():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    q = """
    { viewer { homes { consumption(resolution:HOURLY,last:48) {
      nodes { from consumption cost }
    }}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": q}, headers=hdr)
    j = r.json()
    if "data" not in j or "errors" in j:
        return []
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_consumption(cons):
    yd = datetime.date.today() - datetime.timedelta(days=1)
    out = []
    for r in cons:
        try:
            d = datetime.datetime.fromisoformat(r['from']).astimezone(local_tz).date()
            if d == yd: out.append(r)
        except:
            pass
    return out

def prepare_price_info(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur_dt = datetime.datetime.fromisoformat(pd['current']['startsAt']).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    # Zukunft
    slots = [(datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz), s['total']*100)
             for s in pd['today']+pd['tomorrow']]
    future = [(dt,val) for dt,val in slots if dt>=cur_dt]
    if future:
        ft,fv = min(future, key=lambda x: x[1])
        hours_to_lowest = round((ft-cur_dt).total_seconds()/3600)
        lowest_future_val = fv
    else:
        hours_to_lowest, lowest_future_val = 0, 0
    return {
        "current_price":    cur_price,
        "lowest_today":     lowest_today,
        "highest_today":    highest_today,
        "hours_to_lowest":  hours_to_lowest,
        "lowest_future_val":lowest_future_val
    }

# ——— PV‐Daten aus dem lokalen Docker‐Service ———
def get_historical_pv():
    url = f"{API_BASE}/appGetHistPower/{DTU_SN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    raw = r.json()
    powers = raw.get("power", [])
    times  = raw.get("time", [])
    data = []
    for p,t in zip(powers, times):
        ts = datetime.datetime.fromtimestamp(t.get("seconds",0), tz=local_tz)
        nanos = t.get("nanos", 0)
        if nanos: ts += datetime.timedelta(microseconds=nanos/1000)
        data.append({"startsAt": ts.isoformat(), "power": p})
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    left  = [e for e in data if datetime.datetime.fromisoformat(e["startsAt"]).date()==yesterday]
    right = [e for e in data if datetime.datetime.fromisoformat(e["startsAt"]).date()==today]
    return left, right

# ——— Drawing‐Funktionen ———
def draw_pv_chart(d, left, right, fonts):
    X0,X1 = 60,800; Y0,Y1 =   0,160; W,H = X1-X0, Y1-Y0; PW = W/2
    vals_l = [e["power"] for e in left]; vals_r = [e["power"] for e in right]
    vmin = 0; vmax = max(vals_l+vals_r+[1]); sy = H/(vmax-vmin)
    # Y‐Axis
    step = max(vmax//5,1); yv = 0
    while yv <= vmax:
        y = Y1 - (yv-vmin)*sy
        d.line((X0,y,X1,y),fill=0)
        d.text((X0-45,y-7), f"{int(yv)}W", font=fonts["small"], fill=0)
        yv += step
    d.text((X0, Y0+5), "PV-Erzeugung (W)", font=fonts["small"], fill=0)
    # Gestern
    if len(vals_l)>1:
        xL = [X0 + i*(PW/(len(vals_l)-1)) for i in range(len(vals_l))]
        for i in range(len(vals_l)-1):
            d.line(
                (xL[i],   Y1-(vals_l[i]-vmin)*sy,
                 xL[i+1], Y1-(vals_l[i+1]-vmin)*sy),
                fill=0, width=2
            )
    # Heute
    if len(vals_r)>1:
        xR = [X0+PW + i*(PW/(len(vals_r)-1)) for i in range(len(vals_r))]
        for i in range(len(vals_r)-1):
            d.line(
                (xR[i],   Y1-(vals_r[i]-vmin)*sy,
                 xR[i+1], Y1-(vals_r[i+1]-vmin)*sy),
                fill=0, width=2
            )
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    d.text((X0+10, Y1+5),     "Gestern", font=bf, fill=0)
    d.text((X0+PW+10, Y1+5),   "Heute",   font=bf, fill=0)

def draw_dashed_line(d, x1,y1, x2,y2, **kw):
    dx,dy = x2-x1, y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get("dash_length",4), kw.get("gap_length",4)
    step = dl+gl
    for i in range(int(dist/step)+1):
        s = i*step; e = min(s+dl, dist)
        rs, re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get("fill",0), width=kw.get("width",1))

def draw_two_day_price_chart(d, left_data, right_data, fonts, mode):
    X0,X1 = 60,800; Y0,Y1 = 160,480; W,H = X1-X0, Y1-Y0; PW = W/2
    vals_l = [s['total']*100 for s in left_data]
    vals_r = [s['total']*100 for s in right_data]
    allv   = vals_l+vals_r
    vmin,vmax = (min(allv)-0.5, max(allv)+0.5) if allv else (0,1)
    sy = H/(vmax-vmin)
    # Y‐Axis
    step = 5; yv = math.floor(vmin/step)*step
    while yv <= vmax:
        y = Y1 - (yv-vmin)*sy
        d.line((X0-5,y,X0,y), fill=0); d.line((X1,y,X1+5,y), fill=0)
        d.text((X0-45,y-7), f"{yv/100:.2f}", font=fonts["small"], fill=0)
        yv += step
    d.text((X0-45, Y0-20), "Preis (ct/kWh)", font=fonts["small"], fill=0)
    # Gestern
    times_l = [datetime.datetime.fromisoformat(s["startsAt"]).astimezone(local_tz) for s in left_data]
    if len(times_l)>1:
        xL = [X0 + i*(PW/(len(times_l)-1)) for i in range(len(times_l))]
        for i in range(len(times_l)-1):
            d.line((xL[i], Y1-(vals_l[i]-vmin)*sy, xL[i+1], Y1-(vals_l[i+1]-vmin)*sy),
                   fill=0, width=2)
    # Heute
    times_r = [datetime.datetime.fromisoformat(s["startsAt"]).astimezone(local_tz) for s in right_data]
    if len(times_r)>1:
        xR = [X0+PW + i*(PW/(len(times_r)-1)) for i in range(len(times_r))]
        for i in range(len(times_r)-1):
            d.line((xR[i], Y1-(vals_r[i]-vmin)*sy, xR[i+1], Y1-(vals_r[i+1]-vmin)*sy),
                   fill=0, width=2)
    # Trenner
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

def draw_subtitle_labels(d, fonts):
    X0,X1 = 60,800; PW = (X1-X0)/2; y = 495
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    d.text((X0+10, y),     "Preise gestern", font=bf, fill=0)
    d.text((X0+PW+10, y),  "Preise heute",   font=bf, fill=0)

def draw_info_box(d, info, fonts):
    X0,X1 = 60,800; y = 515
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    infos = [
      f"Aktueller Preis: {info['current_price']/100:.2f}",
      f"Tagestief:       {info['lowest_today']/100:.2f}",
      f"Tageshoch:       {info['highest_today']/100:.2f}",
      f"Tiefstpreis in:  {info['hours_to_lowest']}h | {info['lowest_future_val']/100:.2f}"
    ]
    w = (X1-X0)/len(infos)
    for i,t in enumerate(infos):
        d.text((X0+i*w+5, y), t, font=bf, fill=0)

# ——— Hauptprogramm ———
def main():
    epd = epd7in5_V2.EPD(); epd.init(); epd.Clear()
    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        "small": ImageFont.load_default(),
        "info":  ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    # PV‐Chart (oben)
    left_pv, right_pv = get_historical_pv()
    draw_pv_chart(d, left_pv, right_pv, fonts)

    # Preis‐Chart (unten)
    pd = get_price_data(); update_price_cache(pd)
    cy = get_cached_yesterday_price()
    info = prepare_price_info(pd)
    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode = 'future'
        left_price  = pd['today']
        right_price = pd['tomorrow']
    else:
        mode = 'historical'
        left_price  = cy
        right_price = pd['today']
    draw_two_day_price_chart(d, left_price, right_price, fonts, mode)
    draw_subtitle_labels(d, fonts)
    draw_info_box(d, info, fonts)

    # Update‐Zeit
    d.text((10, 10), time.strftime("Update: %H:%M %d.%m.%Y"), font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__=="__main__":
    main()