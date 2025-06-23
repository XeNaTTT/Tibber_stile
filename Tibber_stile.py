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

# Zeitzone (Python 3.9+ bzw. backports)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare E-Paper–Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# --- Caching-Konstanten ---
CACHE_FILE_TODAY     = 'cached_today_price.json'
CACHE_FILE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    return None

def update_price_cache(price_data):
    today_str = datetime.date.today().isoformat()
    cached = load_cache(CACHE_FILE_TODAY)
    if not cached or cached.get('date') != today_str:
        if cached:
            save_cache(cached, CACHE_FILE_YESTERDAY)
        save_cache({ "date": today_str, "data": price_data['today'] }, CACHE_FILE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    headers = {
        "Authorization": "Bearer " + api_key.API_KEY,
        "Content-Type": "application/json"
    }
    query = """
    {
      viewer {
        homes {
          currentSubscription {
            priceInfo {
              today    { total startsAt }
              tomorrow { total startsAt }
              current  { total startsAt }
            }
          }
        }
      }
    }
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={ "query": query },
                      headers=headers)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def get_consumption_data():
    headers = {
        "Authorization": "Bearer " + api_key.API_KEY,
        "Content-Type": "application/json"
    }
    query = """
    {
      viewer {
        homes {
          consumption(resolution: HOURLY, last: 48) {
            nodes { from consumption cost }
          }
        }
      }
    }
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={ "query": query },
                      headers=headers)
    j = r.json()
    if "data" not in j or "errors" in j:
        return None
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_consumption(cons_data):
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    out = []
    if not cons_data:
        return out
    for rec in cons_data:
        try:
            dt = datetime.datetime.fromisoformat(rec['from']).astimezone(local_tz).date()
            if dt == yesterday:
                out.append(rec)
        except:
            pass
    return out

def draw_dashed_line(draw, x1, y1, x2, y2, fill=0, width=1, dash_length=4, gap_length=4):
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    step = dash_length + gap_length
    for i in range(int(dist/step) + 1):
        start = i * step
        end = min(start + dash_length, dist)
        rs, re = start/dist, end/dist
        sx, sy = x1 + dx*rs, y1 + dy*rs
        ex, ey = x1 + dx*re, y1 + dy*re
        draw.line((sx, sy, ex, ey), fill=fill, width=width)

def draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode):
    # Chart-Bounds
    X0, X1 = 60, 800
    Y0, Y1 = 50, 400
    W, H    = X1 - X0, Y1 - Y0
    PW      = W / 2

    # Sammle Preisdaten
    vals_l = [s['total'] * 100 for s in left_data]
    vals_r = [s['total'] * 100 for s in right_data]
    all_vals = vals_l + vals_r
    if all_vals:
        vmin, vmax = min(all_vals) - 0.5, max(all_vals) + 0.5
    else:
        vmin, vmax = 0, 1
    scale_y = H / (vmax - vmin)

    # Y-Achse & Ticks
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        y = Y1 - (yv - vmin) * scale_y
        # linke und rechte Tick
        draw.line((X0-5, y, X0, y), fill=0)
        draw.line((X1,   y, X1+5, y), fill=0)
        draw.text((X0-45, y-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    draw.text((X0-45, Y0-20), "Preis (ct/kWh)", font=fonts['small'], fill=0)

    # Linkes Panel: Preisdaten zeichnen
    times_l = [
        datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
        for s in left_data
    ]
    nL = len(times_l)
    xL = [X0 + i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]
    for i in range(nL-1):
        x1, y1 = xL[i],   Y1 - (vals_l[i] - vmin) * scale_y
        x2, y2 = xL[i+1], Y1 - (vals_l[i+1] - vmin) * scale_y
        draw.line((x1, y1, x2, y1), fill=0, width=2)
        draw.line((x2, y1, x2, y2), fill=0, width=2)

    # Verbrauch im historical-Modus (nur wenn echte Daten vorhanden)
    if mode == 'historical':
        cons_all = get_consumption_data()
        cons_yesterday = filter_yesterday_consumption(cons_all)
        if cons_yesterday and len(cons_yesterday) == nL:
            cons_vals = [r['consumption'] for r in cons_yesterday]
            cmin, cmax = min(cons_vals), max(cons_vals)
            span = cmax - cmin if cmax > cmin else 1
            y_cons = [Y1 - ((c - cmin)/span) * H for c in cons_vals]
            for i in range(len(y_cons)-1):
                draw_dashed_line(
                    draw,
                    xL[i],   y_cons[i],
                    xL[i+1], y_cons[i+1],
                    fill=0, width=1, dash_length=4, gap_length=4
                )

    # X-Achse-Beschriftung links
    for i, dt in enumerate(times_l):
        if i % 2 == 0:
            draw.text((xL[i], Y1+5), dt.strftime("%Hh"), font=fonts['small'], fill=0)

    # Rechtes Panel: Preisdaten zeichnen
    times_r = [
        datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
        for s in right_data
    ]
    nR = len(times_r)
    xR = [X0+PW + i*(PW/(nR-1)) for i in range(nR)] if nR>1 else [X0+PW]
    for i in range(nR-1):
        x1, y1 = xR[i],   Y1 - (vals_r[i] - vmin) * scale_y
        x2, y2 = xR[i+1], Y1 - (vals_r[i+1] - vmin) * scale_y
        draw.line((x1, y1, x2, y1), fill=0, width=2)
        draw.line((x2, y1, x2, y2), fill=0, width=2)

    # X-Achse-Beschriftung rechts
    for i, dt in enumerate(times_r):
        if i % 2 == 0:
            draw.text((xR[i], Y1+5), dt.strftime("%Hh"), font=fonts['small'], fill=0)

    # Preise bei zwei höchsten und einem niedrigsten Preisstufen des aktuellen Tages
    if vals_r:
        sorted_i = sorted(range(nR), key=lambda i: vals_r[i])
        idx_min    = sorted_i[0]
        idx_high1  = sorted_i[-1]
        idx_high2  = sorted_i[-2] if nR>1 else idx_high1
        for idx, offset in ((idx_min, 40), (idx_high1, 15), (idx_high2, 15)):
            x = xR[idx]
            y = Y1 - (vals_r[idx] - vmin) * scale_y
            txt = f"{vals_r[idx]/100:.2f}"
            draw.text((x-15, y-offset), txt, font=fonts['small'], fill=0)

    # Trenner in der Mitte
    draw.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

    # Aktueller Stunden-Marker
    now = datetime.datetime.now(local_tz)
    def locate(times, vals, xs):
        for i in range(len(times)-1):
            if times[i] <= now < times[i+1]:
                frac = (now - times[i]).total_seconds() / (times[i+1] - times[i]).total_seconds()
                return xs[i] + frac*(xs[i+1] - xs[i]), Y1 - (vals[i] - vmin)*scale_y, i
        return xs[-1], Y1 - (vals[-1] - vmin)*scale_y, len(vals)-1

    if mode == 'future':
        xm, ym, idx = locate(times_l, vals_l, xL)
    else:
        xm, ym, idx = locate(times_r, vals_r, xR)

    r = 5
    draw.ellipse((xm-r, ym-r, xm+r, ym+r), fill=0)
    price = (vals_l if mode=='future' else vals_r)[idx] / 100.0
    draw.text((xm+8, ym-8), f"{price:.2f}", font=fonts['small'], fill=0)

def draw_subtitle_labels(draw, fonts, mode):
    X0, X1 = 60, 800
    PW = (X1 - X0) / 2
    y = 415
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    if mode == "future":
        draw.text((X0+10, y),       "Preis heute", font=bf, fill=0)
        draw.text((X0+PW+10, y),    "Preis morgen", font=bf, fill=0)
    else:
        draw.text((X0+10, y),       "Preise gestern", font=bf, fill=0)
        draw.text((X0+PW+10, y),    "Preis heute",   font=bf, fill=0)

def draw_info_box(draw, data, fonts):
    X0, X1 = 60, 800
    y = 440
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    texts = [
        f"Aktueller Preis: {data['current_price']/100:.2f}",
        f"Tagestief:       {data['lowest_today']/100:.2f}",
        f"Tageshoch:       {data['highest_today']/100:.2f}",
        f"Tiefstpreis in:  {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    w = (X1 - X0) / len(texts)
    for i, txt in enumerate(texts):
        draw.text((X0 + i*w + 5, y), txt, font=bf, fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    draw = ImageDraw.Draw(img)
    fonts = {
        "small":     ImageFont.load_default(),
        "info_font": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    }

    price_data = get_price_data()
    update_price_cache(price_data)
    cache_yest = get_cached_yesterday()

    if price_data['tomorrow'] and price_data['tomorrow'][0]['total'] is not None:
        mode = 'future'
        left_data,  left_type  = price_data['today'],    "price"
        right_data, right_type = price_data['tomorrow'], "price"
    else:
        mode = 'historical'
        ydata = cache_yest.get('data', []) if cache_yest else []
        # Build yesterday slots with 'startsAt' to match format
        left_data = [
            {
              "startsAt": rec.get('startsAt', rec['from']),
              "total":    rec.get('total', 0),
              "consumption": rec.get('consumption', 0)
            }
            for rec in ydata
        ]
        left_type, right_data, right_type = "price", price_data['today'], "price"

    draw_two_day_chart(draw,  left_data, left_type,
                       right_data, right_type,
                       fonts, mode)
    draw_subtitle_labels(draw, fonts, mode)

    info = {
        "current_price":    price_data['current']['total'] * 100,
        "lowest_today":     min([s['total']*100 for s in price_data['today']]),
        "highest_today":    max([s['total']*100 for s in price_data['today']]),
        "hours_to_lowest":  0,
        "lowest_future_val": 0
    }
    draw_info_box(draw, info, fonts)
    draw.text((10, 470),
              time.strftime("Update: %H:%M %d.%m.%Y"),
              font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()
    time.sleep(30)