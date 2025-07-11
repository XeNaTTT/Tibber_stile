#!/usr/bin/python3
# -*- coding:utf-8 -*-

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

# Pfade zum Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# ----- TIBBER-Preis-Cache -----
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
        if ct: save_cache(ct, CACHE_FILE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_FILE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
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

def get_consumption_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes { consumption(resolution:HOURLY,last:48) {
      nodes { from consumption cost }
    }}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query":q}, headers=hdr)
    j = r.json()
    if "data" not in j or "errors" in j:
        return None
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_consumption(cons):
    yd = datetime.date.today() - datetime.timedelta(days=1)
    out=[]
    if not cons: return out
    for r in cons:
        try:
            d = datetime.datetime.fromisoformat(r['from']).astimezone(local_tz).date()
            if d==yd: out.append(r)
        except: pass
    return out

def prepare_data(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur_dt = datetime.datetime.fromisoformat(pd['current']['startsAt']).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    all_slots = [(datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz),
                  s['total']*100) for s in pd['today']+pd['tomorrow']]
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

# ----- PRICE CHART DRAWING (dein Original) -----
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

def draw_two_day_chart(d, left_data, lt, right_data, rt, fonts, mode):
    # ... (dein kompletter draw_two_day_chart-Block hier unverändert) ...
    pass

def draw_subtitle_labels(d,fonts,mode):
    # ... also unverändert ...
    pass

def draw_info_box(d,data,fonts):
    # ... unverändert ...
    pass

# ----- PV-CHART aus SQLite direkt mit PIL -----
DB_FILE = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"

def load_pv_from_db(date):
    import pandas as pd
    start = datetime.datetime.combine(date, datetime.time.min).timestamp()
    end   = datetime.datetime.combine(date, datetime.time.max).timestamp()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT ts, pv_power FROM pv_log WHERE ts BETWEEN ? AND ? ORDER BY ts",
        conn,
        params=(start,end),
        parse_dates=["ts"],
        date_parser=lambda x: datetime.datetime.fromtimestamp(int(x))
    )
    conn.close()
    if df.empty: return []
    df.set_index("ts",inplace=True)
    df = df.resample("15T").mean().fillna(0)
    return list(df.itertuples(index=True, name=None))

def draw_pv_panels(d, fonts):
    # lade Daten
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    data_y = load_pv_from_db(yesterday)
    data_t = load_pv_from_db(today)

    # Panel-Koordinaten
    X0,Y0,X1,Y1 = 60, 420, 800, 780  # unter dem Preisbereich
    W = X1-X0; H = Y1-Y0; PW = W//2

    # Achse Y skalieren über beide Panels
    all_vals = [v for _,v in data_y] + [v for _,v in data_t]
    if all_vals:
        vmin,vmax = min(all_vals), max(all_vals)
    else:
        vmin,vmax = 0,1
    sy = H/(vmax-vmin) if vmax>vmin else 1

    # Funktion zum Zeichnen eines Subpanels
    def draw_panel(data, ox, title):
        # Linie
        pts = []
        for i,(ts,val) in enumerate(data):
            x = ox + (i*(PW/(len(data)-1)) if len(data)>1 else PW/2)
            y = Y1 - (val-vmin)*sy
            pts.append((x,y))
        for p1,p2 in zip(pts,pts[1:]):
            d.line((*p1,p2[0],p1[1]),fill=0,width=2)
            d.line((p2[0],p1[1],*p2),fill=0,width=2)
        # Titel
        d.text((ox, Y0-20), title, font=fonts["small"], fill=0)
        # X-Ticks: jede 2 Stunden
        for h in range(0,25,2):
            frac = h/24
            xt = ox + frac*PW
            d.line((xt,Y1,xt,Y1+5),fill=0)
            d.text((xt-15, Y1+8), f"{h:02d}:00", font=fonts["small"], fill=0)
        # Trenner
        d.line((ox+PW, Y0, ox+PW, Y1), fill=0)

    draw_panel(data_y, X0, f"PV {yesterday.isoformat()}")
    draw_panel(data_t, X0+PW, f"PV {today.isoformat()}")

# ----- MAIN -----
def main():
    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
        "small": ImageFont.load_default(),
        "info_font": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    }

    # 1) Preis-Chart
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
        left_data = [{"startsAt":r.get('startsAt',r.get('from')),"total":r.get('total',0)} for r in ydata]
        right_data = pd['today']

    draw_two_day_chart(d, left_data, "price", right_data, "price", fonts, mode)
    draw_subtitle_labels(d, fonts, mode)
    draw_info_box(d, info, fonts)

    # 2) PV-Chart direkt unterhalb zeichnen
    draw_pv_panels(d, fonts)

    # Footer
    now = datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, epd.height-20), now, font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)

if __name__=="__main__":
    main()
