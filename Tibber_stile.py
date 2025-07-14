#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, time, math, json, requests, datetime, sqlite3
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Waveshare-Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key
DB_FILE = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'

# ---- Tibber Preis-Cache & Abfrage ----
CACHE_TODAY     = 'cached_today_price.json'
CACHE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data, f)

def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def get_price_data():
    hdr = {"Authorization": "Bearer " + api_key.API_KEY,
           "Content-Type": "application/json"}
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query": q}, headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pd['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY)

def prepare_data(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest     = min(today_vals) if today_vals else 0
    highest    = max(today_vals) if today_vals else 0
    cur_dt     = datetime.datetime.fromisoformat(
        pd['current']['startsAt']).astimezone(local_tz)
    cur_price  = pd['current']['total']*100
    slots      = [
        (datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz),
         s['total']*100)
        for s in pd['today'] + pd['tomorrow']
    ]
    future = [(dt,val) for dt,val in slots if dt >= cur_dt]
    if future:
        ft,fv = min(future, key=lambda x: x[1])
        hours  = round((ft - cur_dt).total_seconds()/3600)
    else:
        hours,fv = 0,0
    return {
        "current_price":    cur_price,
        "lowest_today":     lowest,
        "highest_today":    highest,
        "hours_to_lowest":  hours,
        "lowest_future_val":fv
    }

# ---- PV Series Alignment ----
def get_pv_series(slots):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT ts,dtu_power FROM pv_log", conn)
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0)
    df.index = df.index.tz_localize(local_tz)
    vals = []
    for s in slots:
        dt = datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz)
        v  = df['dtu_power'].asof(dt)
        vals.append(float(v) if not pd.isna(v) else 0.0)
    return pd.Series(vals)

# ---- Chart-Helfer mit PV-Overlay und Peak-Labels ----
def draw_dashed_line(d, x1,y1,x2,y2, **kw):
    dx,dy = x2-x1, y2-y1
    dist  = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get('dash_length',4), kw.get('gap_length',4)
    step  = dl+gl
    for i in range(int(dist/step)+1):
        s,e = i*step, min(i*step+dl, dist)
        rs,re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get('fill',0), width=kw.get('width',1))


def draw_two_day_chart(d, left, right, fonts, mode, area, pv_y=None, pv_t=None):
    # area defines drawing box
    X0, Y0, X1, Y1 = area
    W = X1 - X0; H = Y1 - Y0; PW = W/2

    # Preis-Skalierung
    vals_l = [s['total']*100 for s in left]
    vals_r = [s['total']*100 for s in right]
    allp = vals_l + vals_r
    if allp:
        vmin_p, vmax_p = min(allp)-0.5, max(allp)+0.5
    else:
        vmin_p, vmax_p = 0,1
    sy_p = H/(vmax_p - vmin_p)

    # PV-Skalierung
    if pv_y is not None and pv_t is not None:
        pm = max(pv_y.max(), pv_t.max(), 0)
        sy_v = H/(pm+20) if pm>0 else None
    else:
        sy_v = None

    # Y-Achse Preis
    step = 5
    yv = math.floor(vmin_p/step)*step
    while yv <= vmax_p:
        y = Y1 - (yv-vmin_p)*sy_p
        d.line((X0-5, y, X0, y), fill=0)
        d.line((X1, y, X1+5, y), fill=0)
        d.text((X0-45, y-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    d.text((X0-45, Y0-20), 'Preis (ct/kWh)', font=fonts['small'], fill=0)

    # Linkes Panel
    times_l = [datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz) for s in left]
    nL = len(times_l)
    xL = [X0 + i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]
    # Preis-Stufen zeichnen
    for i in range(nL-1):
        x1,y1 = xL[i],   Y1 - (vals_l[i]   - vmin_p)*sy_p
        x2,y2 = xL[i+1], Y1 - (vals_l[i+1] - vmin_p)*sy_p
        d.line((x1,y1,x2,y1), fill=0, width=2)
        d.line((x2,y1,x2,y2), fill=0, width=2)
    # PV-Overlay gestern
    if sy_v:
        pts=[]
        for i,v in enumerate(pv_y.tolist()): pts.append((xL[i], Y1-int(v*sy_v)))
        for a,b in zip(pts, pts[1:]): draw_dashed_line(d,a[0],a[1],b[0],b[1],dash_length=2,gap_length=2)
        # Peak-Label
        idx_max = pv_y.values.argmax()
        xm, ym = pts[idx_max]
        d.text((xm-10, ym-15), f"{int(pv_y[idx_max])}W", font=fonts['small'], fill=0)
    # X-Ticks
    for i,dt in enumerate(times_l):
        if i%2==0: d.text((xL[i],Y1+5), dt.strftime('%Hh'), font=fonts['small'], fill=0)

    # Mitte
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

    # Rechtes Panel
    times_r = [datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz) for s in right]
    nR = len(times_r)
    xR = [X0+PW + i*(PW/(nR-1)) for i in range(nR)] if nR>1 else [X0+PW]
    for i in range(nR-1):
        x1,y1 = xR[i],   Y1 - (vals_r[i]   - vmin_p)*sy_p
        x2,y2 = xR[i+1], Y1 - (vals_r[i+1] - vmin_p)*sy_p
        d.line((x1,y1,x2,y1), fill=0, width=2)
        d.line((x2,y1,x2,y2), fill=0, width=2)
    # PV-Overlay heute
    if sy_v:
        pts=[]
        for i,v in enumerate(pv_t.tolist()): pts.append((xR[i], Y1-int(v*sy_v)))
        for a,b in zip(pts, pts[1:]): draw_dashed_line(d,a[0],a[1],b[0],b[1],dash_length=2,gap_length=2)
        idx_max = pv_t.values.argmax()
        xm, ym = pts[idx_max]
        d.text((xm-10, ym-15), f"{int(pv_t[idx_max])}W", font=fonts['small'], fill=0)
    # X-Ticks
    for i,dt in enumerate(times_r):
        if i%2==0: d.text((xR[i],Y1+5), dt.strftime('%Hh'), font=fonts['small'], fill=0)


def draw_subtitle_labels(d, fonts, mode):
    bf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',12)
    X0,X1 = 60, epd7in5_V2.EPD().width; PW=(X1-X0)/2; y=epd7in5_V2.EPD().height//2 - 5
    d.text((X0+10,y),     'Preise & PV gestern', font=bf, fill=0)
    d.text((X0+PW+10,y), 'Preis & PV heute',    font=bf, fill=0)


def draw_info_box(d, info, fonts):
    X0,X1 = 60, epd7in5_V2.EPD().width; y=epd7in5_V2.EPD().height//2 + 20
    bf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',12)
    infos = [
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tagestief:   {info['lowest_today']/100:.2f}",
        f"Tageshoch:   {info['highest_today']/100:.2f}",
        f"Tiefstpreis in {info['hours_to_lowest']}h"
    ]
    w = (X1-X0)/len(infos)
    for i,t in enumerate(infos): d.text((X0+i*w+5,y), t, font=bf, fill=0)

# ---- Main ----
def main():
    epd = epd7in5_V2.EPD(); epd.init(); epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = { 'small':ImageFont.load_default(),
              'info_font':ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14) }

    # Preisdaten
    pinfo = get_price_data(); update_price_cache(pinfo)
    cy    = get_cached_yesterday(); info = prepare_data(pinfo)
    left_price  = cy.get('data', []);
    right_price = pinfo['today']

    # PV Serien
    pv_y = get_pv_series(left_price)
    pv_t = get_pv_series(right_price)
    print("PV gestern:", pv_y.tolist())
    print("PV heute: ", pv_t.tolist())

    # Kombiniertes Chart oben mit Margin
    left_margin = 60
    right_margin = 10
    upper = (left_margin, 0, epd.width-right_margin, epd.height//2)
    draw_two_day_chart(d, left_price, right_price, fonts, 'historical', upper, pv_y, pv_t)

    draw_subtitle_labels(d, fonts, 'historical')
    draw_info_box(d, info, fonts)

    # Footer
    now_str = datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, epd.height-20), now_str, font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep(); time.sleep(30)

if __name__=='__main__':
    main()
