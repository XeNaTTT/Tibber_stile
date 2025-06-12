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

# ZoneInfo für Zeitzonenkonversion (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# E-Paper–Treiberpfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2
import api_key

# Caching
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
        save_cache({"date": today_str, "data": price_data['today']}, CACHE_FILE_TODAY)

def get_cached_yesterday_price():
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
              today { total startsAt }
              tomorrow { total startsAt }
              current  { total startsAt }
            }
          }
        }
      }
    }
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query": query},
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
                      json={"query": query},
                      headers=headers)
    j = r.json()
    if "data" not in j or "errors" in j:
        return None
    try:
        return j['data']['viewer']['homes'][0]['consumption']['nodes']
    except Exception:
        return None

def filter_yesterday_consumption(consumption_data):
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    out = []
    if not consumption_data:
        return out
    for rec in consumption_data:
        try:
            dt = datetime.datetime.fromisoformat(rec['from']).astimezone(local_tz).date()
            if dt == yesterday:
                out.append(rec)
        except Exception:
            pass
    return out

def prepare_data(price_data):
    timestamps, labels, prices = [], [], []
    for slot in price_data['today']:
        dt = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
        timestamps.append(dt)
        labels.append(dt.strftime("%d.%m %Hh"))
        prices.append(slot['total']*100)
    return {
        "timestamps": timestamps,
        "labels":     labels,
        "prices":     prices,
        "current_price": price_data['current']['total']*100,
        "lowest_today":  min([s['total']*100 for s in price_data['today']]),
        "highest_today": max([s['total']*100 for s in price_data['today']]),
        "hours_to_lowest": 0, "lowest_future_val":0
    }

def draw_dashed_line(draw, x1, y1, x2, y2, fill=0, width=1, dash_length=4, gap_length=4):
    dx, dy = x2-x1, y2-y1
    dist = math.hypot(dx, dy)
    if dist==0: return
    step = dash_length+gap_length
    for i in range(int(dist/step)+1):
        start = i*step
        end = min(start+dash_length, dist)
        rs, re = start/dist, end/dist
        sx, sy = x1+dx*rs, y1+dy*rs
        ex, ey = x1+dx*re, y1+dy*re
        draw.line((sx,sy,ex,ey), fill=fill, width=width)

def draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode):
    X0, X1 = 60, 800
    Y0, Y1 = 50, 400
    W, H   = X1-X0, Y1-Y0
    PW     = W/2

    # Linkes Panel: immer Preisdaten aus left_data
    times_left  = []
    values_left = []
    for slot in left_data:
        dt = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
        times_left.append(dt)
        values_left.append(slot['total']*100)
    nL = len(values_left)

    # X-Positionen links
    if nL>1:
        xL = [X0 + i*(PW/(nL-1)) for i in range(nL)]
    else:
        xL = [X0]

    # Y-Skalierung über beide Panels
    all_vals = values_left + [s['total']*100 for s in right_data]
    if all_vals:
        vmin = min(all_vals)-0.5
        vmax = max(all_vals)+0.5
    else:
        vmin, vmax = 0,1
    scale_y = H/(vmax-vmin)

    # Preislinie links
    for i in range(nL-1):
        x1, y1 = xL[i], Y1-(values_left[i]-vmin)*scale_y
        x2, y2 = xL[i+1], Y1-(values_left[i+1]-vmin)*scale_y
        draw.line((x1,y1,x2,y1), fill=0, width=2)
        draw.line((x2,y1,x2,y2), fill=0, width=2)

    # Verbrauchslinie im historischen Modus
    if mode=="historical":
        cons = [slot.get('consumption') for slot in left_data]
        if cons and len(cons)==nL and any(c is not None for c in cons):
            cmn = min(cons); cmx = max(cons)
            span = cmx-cmn if cmx>cmn else 1
            y_cons = [Y1 - ((c-cmn)/span)*H for c in cons]
            for i in range(len(y_cons)-1):
                draw_dashed_line(draw, xL[i], y_cons[i], xL[i+1], y_cons[i+1],
                                 fill=0,width=1,dash_length=4,gap_length=4)

    # X-Achse-Beschriftung links
    for i, dt in enumerate(times_left):
        if i%2==0:
            draw.text((xL[i], Y1+5), dt.strftime("%Hh"), font=fonts["small"], fill=0)

    # Mittlerer Trenner
    draw.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

    # (rechtes Panel wie gehabt …)

def draw_subtitle_labels(draw, fonts, mode):
    X0, X1 = 60, 800
    PW = (X1-X0)/2
    lbl_y = 415
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    if mode=="future":
        draw.text((X0+10, lbl_y),"Preis heute", font=bf, fill=0)
        draw.text((X0+PW+10, lbl_y),"Preis morgen", font=bf, fill=0)
    else:
        draw.text((X0+10, lbl_y),"Verbrauch gestern", font=bf, fill=0)
        draw.text((X0+PW+10,lbl_y),"Preis heute", font=bf, fill=0)

def draw_info_box(draw, data, fonts):
    X0, X1 = 60,800
    Y = 440
    bf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    infos = [
        f"Aktueller Preis: {data['current_price']/100:.2f}",
        f"Tagestief:       {data['lowest_today']/100:.2f}",
        f"Tageshoch:       {data['highest_today']/100:.2f}",
        f"Tiefstpreis in:  {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    w = (X1-X0)/len(infos)
    for i, txt in enumerate(infos):
        draw.text((X0 + i*w +5, Y), txt, font=bf, fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()

    img  = Image.new('1',(epd.width,epd.height),255)
    draw = ImageDraw.Draw(img)
    f_small = ImageFont.load_default()
    f_info  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    fonts = {"small":f_small,"info_font":f_info}

    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday_price()

    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode = 'future'
        left_data, right_data = pd['today'], pd['tomorrow']
        left_type, right_type = "price","price"
    else:
        mode = 'historical'
        cons_data = get_consumption_data()
        filt = filter_yesterday_consumption(cons_data)
        # linke Daten in historical immer Preis+consumption
        left_data = [{"startsAt":slot['from'], "total":0, "consumption":slot['consumption']} for slot in filt]
        right_data = pd['today']
        left_type, right_type = "combo","price"

    draw_two_day_chart(draw,left_data,left_type,right_data,right_type,fonts,mode)
    draw_subtitle_labels(draw,fonts,mode)
    data = prepare_data(pd)
    draw_info_box(draw,data,fonts)
    draw.text((10,470), time.strftime("Update: %H:%M %d.%m.%Y"),
              font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__=="__main__":
    main()
    time.sleep(30)