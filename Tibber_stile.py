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
from PIL import Image, ImageDraw, ImageFont

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Waveshare Paths
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# ---- Tibber Preis-Cache & Query ----
CACHE_FILE_TODAY     = 'cached_today_price.json'
CACHE_FILE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data,f)
def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_FILE_TODAY)
    if not ct or ct.get('date')!=today:
        if ct:
            save_cache(ct, CACHE_FILE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_FILE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,
           "Content-Type":"application/json"}
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query":q}, headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def prepare_data(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur_dt = datetime.datetime.fromisoformat(
                pd['current']['startsAt']).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    all_slots = [(datetime.datetime.fromisoformat(s['startsAt'])\
                    .astimezone(local_tz), s['total']*100)
                 for s in pd['today']+pd['tomorrow']]
    future = [(dt,val) for dt,val in all_slots if dt>=cur_dt]
    if future:
        ft,fv = min(future, key=lambda x:x[1])
        hours_to_lowest = round((ft-cur_dt).total_seconds()/3600)
        lowest_future_val = fv
    else:
        hours_to_lowest = 0
        lowest_future_val = 0
    return {
        "current_price":    cur_price,
        "lowest_today":     lowest_today,
        "highest_today":    highest_today,
        "hours_to_lowest":  hours_to_lowest,
        "lowest_future_val":lowest_future_val
    }

# ---- Preis-Chart Drawing (deine bestehenden Funktionen) ----
def draw_dashed_line(d, x1,y1,x2,y2, **kw):
    dx,dy = x2-x1,y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get("dash_length",4), kw.get("gap_length",4)
    step = dl+gl
    for i in range(int(dist/step)+1):
        s=i*step; e=min(s+dl,dist)
        rs,re = s/dist,e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get("fill",0), width=kw.get("width",1))

def draw_two_day_chart(d, left_data, lt, right_data, rt, fonts, mode,
                       area=None):
    # area = (x0,y0,x1,y1) or default
    if area:
        X0,Y0,X1,Y1 = area
    else:
        X0,X1=60,800; Y0,Y1=50,400
    W,H = X1-X0, Y1-Y0; PW=W/2
    vals_l = [s['total']*100 for s in left_data]
    vals_r = [s['total']*100 for s in right_data]
    allv=vals_l+vals_r
    if allv:
        vmin,vmax = min(allv)-0.5, max(allv)+0.5
    else:
        vmin,vmax = 0,1
    sy = H/(vmax-vmin)

    # Y-Achse
    step=5; yv=math.floor(vmin/step)*step
    while yv<=vmax:
        y=Y1-(yv-vmin)*sy
        d.line((X0-5,y,X0,y),fill=0); d.line((X1,y,X1+5,y),fill=0)
        d.text((X0-45,y-7),f"{yv/100:.2f}",font=fonts["small"],fill=0)
        yv+=step
    d.text((X0-45,Y0-20),"Preis (ct/kWh)",font=fonts["small"],fill=0)

    # linkes Panel
    times_l=[datetime.datetime.fromisoformat(s["startsAt"])\
             .astimezone(local_tz) for s in left_data]
    nL=len(times_l)
    xL=[X0+i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]
    for i in range(nL-1):
        x1,y1=xL[i],Y1-(vals_l[i]-vmin)*sy
        x2,y2=xL[i+1],Y1-(vals_l[i+1]-vmin)*sy
        d.line((x1,y1,x2,y1),fill=0,width=2); d.line((x2,y1,x2,y2),fill=0,width=2)
    for i,dt in enumerate(times_l):
        if i%2==0:
            d.text((xL[i],Y1+5),dt.strftime("%Hh"),font=fonts["small"],fill=0)

    # mittlere Linie
    d.line((X0+PW,Y0,X0+PW,Y1),fill=0,width=2)

    # rechtes Panel
    times_r=[datetime.datetime.fromisoformat(s["startsAt"])\
             .astimezone(local_tz) for s in right_data]
    nR=len(times_r)
    xR=[X0+PW+i*(PW/(nR-1)) for i in range(nR)] if nR>1 else [X0+PW]
    for i in range(nR-1):
        x1,y1=xR[i],Y1-(vals_r[i]-vmin)*sy
        x2,y2=xR[i+1],Y1-(vals_r[i+1]-vmin)*sy
        d.line((x1,y1,x2,y1),fill=0,width=2); d.line((x2,y1,x2,y2),fill=0,width=2)
    for i,dt in enumerate(times_r):
        if i%2==0:
            d.text((xR[i],Y1+5),dt.strftime("%Hh"),font=fonts["small"],fill=0)

def draw_subtitle_labels(d,fonts,mode):
    # dein bestehender Code hier
    pass

def draw_info_box(d,data,fonts):
    # dein bestehender Code hier
    pass

# ---- PV Stacked-Area Chart unten ----
DB_FILE = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"

def draw_pv_stacked(d, fonts, y0, y1, width):
    import pandas as pd
    today = datetime.date.today()
    start_ts = int(datetime.datetime.combine(today, datetime.time.min).timestamp())
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT ts, pv1_power, pv2_power FROM pv_log WHERE ts>=? ORDER BY ts",
        conn, params=(start_ts,)
    )
    conn.close()
    if df.empty:
        return
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().fillna(0)

    h = y1 - y0
    df['stack'] = df['pv1_power'] + df['pv2_power']
    vmax = df['stack'].max() or 1.0

    pts1, pts2 = [], []
    for i,(ts,row) in enumerate(df.iterrows()):
        x = int(i * width/(len(df)-1))
        y1p = y1 - int((row['pv1_power']/vmax)*h)
        y2p = y1 - int((row['stack']    /vmax)*h)
        pts1.append((x, y1p))
        pts2.append((x, y2p))

    # pv1 Fläche (schwarz)
    poly1 = [(0,y1)] + pts1 + [(width-1,y1)]
    d.polygon(poly1, fill=0)
    # pv2 Fläche (weiss ausschneiden)
    poly2 = [(0,y1)] + pts2 + [(width-1,y1)]
    d.polygon(poly2, fill=255)

    # X-Achse: jede Stunde
    for ts in df.index:
        if ts.minute==0:
            idx = df.index.get_loc(ts)
            x = int(idx * width/(len(df)-1))
            d.line((x,y1,x,y1+4), fill=0)
            d.text((x-15,y1+6), ts.strftime("%H:%M"),
                   font=fonts["small"], fill=0)

    # Y-Achse-Labels
    for frac,label in [(0,"0W"),(0.5,f"{vmax/2:.0f}W"),(1,f"{vmax:.0f}W")]:
        y = y1 - int(frac*h)
        d.line((0,y,4,y), fill=0)
        d.text((6,y-8), label, font=fonts["small"], fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        "small": ImageFont.load_default(),
        "info_font": ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    }

    # 1) Preis-Chart oben
    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday()
    info = prepare_data(pd)
    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode = 'future'
        left_data,right_data = pd['today'], pd['tomorrow']
    else:
        mode = 'historical'
        ydata = cy.get('data',[]) if cy else []
        left_data = [{"startsAt":r.get('startsAt',r.get('from')),
                      "total":r.get('total',0)} for r in ydata]
        right_data = pd['today']

    clip_y = epd.height // 2
    draw_two_day_chart(d, left_data, "price",
                       right_data,"price", fonts, mode,
                       area=(0,0,epd.width,clip_y))
    draw_subtitle_labels(d, fonts, mode)
    draw_info_box(d, info, fonts)

    # 2) PV-Stapel-Flächen unten
    draw_pv_stacked(d, fonts, clip_y, epd.height, epd.width)

    # Footer
    now = datetime.datetime.now(local_tz)\
            .strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, epd.height-20), now,
           font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)

if __name__=="__main__":
    main()
