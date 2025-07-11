#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, time, math, json, requests, datetime, sqlite3
from PIL import Image, ImageDraw, ImageFont

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# ---- Tibber Preis-Cache ----
CACHE_TODAY     = 'cached_today_price.json'
CACHE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn, 'w') as f: json.dump(data, f)
def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def get_price_data():
    hdr = {
      "Authorization": "Bearer "+api_key.API_KEY,
      "Content-Type": "application/json"
    }
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

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date')!=today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY)

def prepare_data(pd):
    today_vals = [s['total']*100 for s in pd['today']]
    lowest = min(today_vals) if today_vals else 0
    highest = max(today_vals) if today_vals else 0
    cur_dt = datetime.datetime.fromisoformat(pd['current']['startsAt']).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    slots = [(datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz), s['total']*100)
             for s in pd['today']+pd['tomorrow']]
    future = [(dt,val) for dt,val in slots if dt>=cur_dt]
    if future:
        ft,fv = min(future, key=lambda x:x[1])
        hours = round((ft-cur_dt).total_seconds()/3600)
    else:
        hours, fv = 0,0
    return {
      "current_price":cur_price,
      "lowest_today":lowest,
      "highest_today":highest,
      "hours_to_lowest":hours,
      "lowest_future_val":fv
    }

# ---- Preis-Chart ----
def draw_dashed_line(d, x1,y1,x2,y2, **kw):
    dx,dy = x2-x1, y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get("dash_length",4), kw.get("gap_length",4)
    step = dl+gl
    for i in range(int(dist/step)+1):
        s,e = i*step, min(i*step+dl, dist)
        rs,re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get("fill",0), width=kw.get("width",1))

def draw_two_day_chart(d, left, lt, right, rt, fonts, mode, area=None):
    # unverändert wie gehabt...
    # Zeichnet deinen Preis-Chart in das Rect area oder default
    # (Implementierung analog deinem bisherigen Code)
    pass

def draw_subtitle_labels(d,fonts,mode):
    # dein bisheriger Code
    pass

def draw_info_box(d,data,fonts):
    # dein bisheriger Code
    pass

# ---- PV-Chart als Linien direkt aus SQLite ----
DB_FILE = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"

def draw_pv_lines(d, fonts, y0, y1, width):
    import pandas as pd
    # 1) Daten einlesen
    today = datetime.date.today()
    start = int(datetime.datetime.combine(today,datetime.time.min).timestamp())
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
      "SELECT ts, pv1_power, pv2_power FROM pv_log WHERE ts>=? ORDER BY ts",
      conn, params=(start,)
    )
    conn.close()

    # Debug
    print(f"[DEBUG] PV rows today: {len(df)}")

    # 2) Fallback: wenn leer, generiere 96 Elemente mit null
    if df.empty:
        idx = pd.date_range(start=datetime.datetime.combine(today,datetime.time.min),
                            periods=96, freq='15T')
        df = pd.DataFrame({'pv1_power':0.0,'pv2_power':0.0}, index=idx)
    else:
        # Timestamp konvertieren
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        df.set_index('ts', inplace=True)
        df = df.resample('15T').mean().fillna(0)

    # Gesamt
    df['total'] = df['pv1_power'] + df['pv2_power']

    # 3) Skalierung
    h = y1 - y0
    vmax = df['total'].max() or 1.0

    # 4) Punkte berechnen
    times = list(df.index)
    n = len(times)
    xs = [int(i*width/(n-1)) for i in range(n)]
    ys1 = [y1 - int((v/vmax)*h) for v in df['pv1_power']]
    ys2 = [y1 - int((v/vmax)*h) for v in df['pv2_power']]
    yst = [y1 - int((v/vmax)*h) for v in df['total']]

    # 5) Linien zeichnen: PV1 (solid), PV2 (dashed), Total (dotted)
    # PV1
    for i in range(n-1):
        d.line((xs[i], ys1[i], xs[i+1], ys1[i+1]), fill=0, width=2)
    # PV2 (dash)
    for i in range(n-1):
        draw_dashed_line(d, xs[i], ys2[i], xs[i+1], ys2[i+1],
                         dash_length=2, gap_length=2, fill=0, width=1)
    # Total (thin solid)
    for i in range(n-1):
        d.line((xs[i], yst[i], xs[i+1], yst[i+1]), fill=0, width=1)

    # 6) Achsen
    # X every hour
    for ts in df.index:
        if ts.minute==0:
            i = df.index.get_loc(ts)
            x = xs[i]
            d.line((x,y1,x,y1+4), fill=0)
            d.text((x-12,y1+6), ts.strftime("%H:%M"), font=fonts["small"], fill=0)
    # Y labels
    for frac,label in [(0,"0W"),(0.5,f"{vmax/2:.0f}W"),(1,f"{vmax:.0f}W")]:
        yy = y1 - int(frac*h)
        d.line((0,yy,4,yy), fill=0)
        d.text((6,yy-8), label, font=fonts["small"], fill=0)

# ---- Main ----
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()

    img = Image.new('1', (epd.width, epd.height), 255)
    d   = ImageDraw.Draw(img)
    fonts = {
      "small": ImageFont.load_default(),
      "info_font": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    }

    # 1) Preis oben
    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday()
    info = prepare_data(pd)
    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode = 'future'
        left, right = pd['today'], pd['tomorrow']
    else:
        mode = 'historical'
        yd = cy.get('data',[]) if cy else []
        left = [{"startsAt":r.get('startsAt',r.get('from')),"total":r.get('total',0)} for r in yd]
        right = pd['today']

    clip_y = epd.height//2
    draw_two_day_chart(d, left, "price", right, "price", fonts, mode,
                       area=(0,0,epd.width,clip_y))
    draw_subtitle_labels(d, fonts, mode)
    draw_info_box(d, info, fonts)

    # 2) PV-Linien unten
    draw_pv_lines(d, fonts, clip_y, epd.height, epd.width)

    # Footer
    now = datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, epd.height-20), now, font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()
    time.sleep(30)

if __name__=="__main__":
    main()
