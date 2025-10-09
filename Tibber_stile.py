#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time, math, json, requests, datetime as dt, sqlite3, logging
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, numpy as np

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("Europe/Berlin")

# E-Paper
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# Pfade & Config
DB_FILE         = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'
CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'
ECOFLOW_FALLBACK= '/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_status.json'

logging.basicConfig(level=logging.INFO)

# ---------- Utils ----------
def save_cache(data, fn):
    with open(fn, 'w') as f: json.dump(data, f)

def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def safe_get(d, *path, default=None):
    for k in path:
        if d is None: return default
        d = d.get(k)
    return d if d is not None else default

# ---------- Tibber ----------
def tibber_priceinfo():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}"}
    gql = '''
    { viewer { homes { currentSubscription { priceInfo {
        today    { total startsAt }
        tomorrow { total startsAt }
        current  { total startsAt }
    }}}}}
    '''
    r = requests.post('https://api.tibber.com/v1-beta/gql',
                      json={"query": gql}, headers=hdr, timeout=15)
    r.raise_for_status()
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}

def prepare_info(pi):
    today_vals = [s['total']*100 for s in pi['today']]
    cur = pi['current']
    return {
        'current_dt': dt.datetime.fromisoformat(cur['startsAt']).astimezone(LOCAL_TZ),
        'current_price': cur['total']*100,
        'lowest_today': min(today_vals) if today_vals else 0,
        'highest_today': max(today_vals) if today_vals else 0
    }

# ---------- 15-Min Transformation ----------
def expand_to_15min(slots):
    """Erzeugt 15-Minuten-Raster (4 Punkte je Stunde, Step-Hold)."""
    ts_list, val_list = [], []
    for s in slots:
        start = dt.datetime.fromisoformat(s['startsAt']).astimezone(LOCAL_TZ)
        price = s['total']*100
        for k in range(4):
            ts_list.append(start + dt.timedelta(minutes=15*k))
            val_list.append(price)
    return ts_list, val_list

# ---------- PV & Verbrauch ----------
def series_from_db(table, column, slots_dt):
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(f"SELECT ts, {column} FROM {table}", conn)
    except Exception:
        conn.close(); return pd.Series([0.0]*len(slots_dt))
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index('ts', inplace=True)
    # auf 15 Minuten glätten
    df = df.resample('15T').mean().ffill().fillna(0)
    out = []
    for t in slots_dt:
        v = df[column].asof(t) if not df.empty else 0.0
        out.append(float(0.0 if pd.isna(v) else v))
    return pd.Series(out)

def get_pv_series(slots_dt):
    return series_from_db("pv_log", "dtu_power", slots_dt)

def get_consumption_series(slots_dt):
    # optional: consumption_log(ts, consumption_w)
    return series_from_db("consumption_log", "consumption_w", slots_dt)

# ---------- Wetter (Open-Meteo, keine API-Keys noetig) ----------
def sunshine_hours(lat, lon):
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               "&daily=sunshine_duration&timezone=Europe%2FBerlin")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        sec = safe_get(r.json(), "daily", "sunshine_duration", default=[0])[0]
        return round((sec or 0)/3600, 1)
    except Exception:
        return None

# ---------- EcoFlow (Fallback auf JSON) ----------
def ecoflow_status():
    # Prioritaet: Open API, sonst Fallback JSON
    try:
        token = getattr(api_key, "ECOFLOW_TOKEN", None)
        device_id = getattr(api_key, "ECOFLOW_DEVICE_ID", None)
        if token and device_id:
            # Beispiel-Endpunkt – ggf. an deine Open-API Route anpassen:
            # https://api.ecoflow.com/iot-open/device/queryDeviceQuota
            url = "https://api.ecoflow.com/iot-open/device/queryDeviceQuota"
            hdr = {"Authorization": token, "Content-Type":"application/json"}
            r = requests.post(url, headers=hdr, json={"deviceSn": device_id}, timeout=10)
            r.raise_for_status()
            j = r.json()
            # Mapping anpassen je nach Antwortstruktur:
            soc = safe_get(j, "data", "soc", default=None)
            pow_w = safe_get(j, "data", "power", default=None)
            mode = safe_get(j, "data", "workMode", default=None)
            eta = safe_get(j, "data", "remainMinutes", default=None)
            return dict(soc=soc, power_w=pow_w, mode=mode, eta_min=eta)
    except Exception:
        pass
    # Fallback JSON
    if os.path.exists(ECOFLOW_FALLBACK):
        try:
            with open(ECOFLOW_FALLBACK) as f:
                return json.load(f)
        except Exception:
            pass
    return {"soc": None, "power_w": None, "mode": None, "eta_min": None}

# ---------- Drawing Helpers ----------
def draw_dashed_line(d, x1, y1, x2, y2, dash=2, gap=2, fill=0, width=1):
    dx, dy = x2-x1, y2-y1
    dist = math.hypot(dx, dy)
    if dist == 0: return
    step = dash+gap
    for i in range(int(dist/step)+1):
        s, e = i*step, min(i*step+dash, dist)
        rs, re = s/dist, e/dist
        xa, ya = x1 + dx*rs, y1 + dy*rs
        xb, yb = x1 + dx*re, y1 + dy*re
        d.line((xa, ya, xb, yb), fill=fill, width=width)

def draw_weather_box(d, x, y, w, h, fonts, sun_hours):
    # Rahmen
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    # Sonnen-Icon
    cx, cy, r = x+25, y+25, 10
    d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=0, width=2)
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        d.line((cx+math.cos(rad)*r*1.6, cy+math.sin(rad)*r*1.6,
                cx+math.cos(rad)*r*2.4, cy+math.sin(rad)*r*2.4), fill=0, width=2)
    title = "Wetter"
    d.text((x+60, y+5), title, font=fonts['bold'], fill=0)
    txt = "Sonnenstunden heute: "
    val = f"{sun_hours:.1f} h" if sun_hours is not None else "—"
    d.text((x+60, y+28), txt+val, font=fonts['small'], fill=0)

def minutes_to_hhmm(m):
    if m is None: return "—"
    try:
        m = int(m)
        return f"{m//60:02d}:{m%60:02d} h"
    except: return "—"

def draw_ecoflow_box(d, x, y, w, h, fonts, st):
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    d.text((x+10, y+5), "EcoFlow Stream AC", font=fonts['bold'], fill=0)
    lines = [
        f"SoC: {st['soc']}%" if st.get('soc') is not None else "SoC: —",
        f"Leistung: {int(st['power_w'])} W" if st.get('power_w') is not None else "Leistung: —",
        f"Modus: {st.get('mode') or '—'}",
        f"Restzeit: {minutes_to_hhmm(st.get('eta_min'))}"
    ]
    for i, t in enumerate(lines):
        d.text((x+10, y+28 + i*18), t, font=fonts['small'], fill=0)

def draw_info_box(d, info, fonts, y, width):
    x0 = 10
    items = [
        ("Preis jetzt", info['current_price']/100),
        ("Tief heute", info['lowest_today']/100),
        ("Hoch heute", info['highest_today']/100),
    ]
    colw = width/len(items)
    for i,(k,v) in enumerate(items):
        d.text((x0 + i*colw, y), f"{k}: {v:.2f} ct", font=fonts['bold'], fill=0)

def draw_two_day_chart(d, left, right, fonts, subtitles, area,
                       pv_left=None, pv_right=None,
                       cons_left=None, cons_right=None,
                       cur_dt=None, cur_price=None):
    X0,Y0,X1,Y1 = area
    W,H = X1-X0, Y1-Y0
    PW  = W/2

    # 15-Min Arrays
    tl, vl = expand_to_15min(left)
    tr, vr = expand_to_15min(right)
    if not (vl or vr): return

    allp = vl + vr
    vmin, vmax = min(allp)-0.5, max(allp)+0.5
    sy_price = H/(vmax - vmin if vmax>vmin else 1)

    # zweite Achse fuer Leistung (PV/Verbrauch)
    def vmax_power(series):
        if series is None: return 0
        try: return float(np.nanmax(series)) if len(series)>0 else 0
        except: return 0
    pmax = max(vmax_power(pv_left), vmax_power(pv_right),
               vmax_power(cons_left), vmax_power(cons_right))
    sy_pow = H/((pmax or 1)*1.2)

    # Y-Ticks (Preis)
    step = 5
    yv = math.floor(vmin/step)*step
    while yv <= vmax:
        yy = Y1 - (yv - vmin)*sy_price
        d.line((X0-4,yy, X1+4,yy), fill=255)  # helle Hilfslinie (loeschen)
        d.line((X0,yy, X1,yy), fill=0, width=1)
        d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['tiny'], fill=0)
        yv += step
    d.text((X0-45, Y0-18), 'Preis (ct/kWh)', font=fonts['tiny'], fill=0)

    def panel(ts_list, val_list, pv_list, cons_list, x0):
        n = len(ts_list)
        if n < 2: return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]

        # Preis als Stufenlinie
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (val_list[i]   - vmin)*sy_price
            x2, y2 = xs[i+1], Y1 - (val_list[i+1] - vmin)*sy_price
            d.line((x1,y1, x2,y1), fill=0, width=2)
            d.line((x2,y1, x2,y2), fill=0, width=2)

        # PV (kurzer Dash)
        if pv_list is not None and n == len(pv_list):
            for i in range(n-1):
                y1 = Y1 - pv_list.iloc[i]*sy_pow
                y2 = Y1 - pv_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=2, gap=2, width=1)

        # Verbrauch (laengerer Dash => visuell unterscheidbar)
        if cons_list is not None and n == len(cons_list):
            for i in range(n-1):
                y1 = Y1 - cons_list.iloc[i]*sy_pow
                y2 = Y1 - cons_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=4, gap=3, width=1)

        # Labels min/max Preis
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], Y1 - (val_list[idx]-vmin)*sy_price
            d.text((xi-12, yi-12), f"{val_list[idx]/100:.2f}", font=fonts['tiny'], fill=0)

    # Panels
    panel(tl, vl, pv_left,  cons_left,  X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(tr, vr, pv_right, cons_right, X0+PW)

    # Subtitles
    d.text((X0+5,    Y1+5), subtitles[0], font=fonts['bold'], fill=0)
    d.text((X0+PW+5, Y1+5), subtitles[1], font=fonts['bold'], fill=0)

    # Legende Leistung
    d.text((X1-180, Y0-16), "— — PV   ——  Verbrauch", font=fonts['tiny'], fill=0)

    # Marker fuer aktuelle Stunde
    if cur_dt and cur_price is not None:
        arr = tl if subtitles[0] in ("Heute","Gestern") and cur_dt.date()==tl[0].date() else tr
        x0  = X0 if arr is tl else X0+PW
        if len(arr)>1:
            # Index des 15-Min Slots
            try:
                idx = next(i for i,t in enumerate(arr) if t.hour==cur_dt.hour and t.minute//15==cur_dt.minute//15)
                px = x0 + idx*(PW/(len(arr)-1))
                py = Y1 - (cur_price - vmin)*sy_price
                r = 4
                d.ellipse((px-r, py-r, px+r, py+r), fill=0)
                d.text((px+r+2, py-r-2), f"{cur_price/100:.2f}", font=fonts['tiny'], fill=0)
            except StopIteration:
                pass

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height

    # Daten laden
    pi   = tibber_priceinfo()
    update_price_cache(pi)
    info = prepare_info(pi)

    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left, right = pi['today'], tomorrow
        labels = ("Heute", "Morgen")
    else:
        left, right = cached_yesterday()['data'], pi['today']
        labels = ("Gestern", "Heute")

    # 15-Min Timestamps fuer beide Panels
    tl_dt, _ = expand_to_15min(left)
    tr_dt, _ = expand_to_15min(right)

    pv_left   = get_pv_series(tl_dt)
    pv_right  = get_pv_series(tr_dt)
    cons_left = get_consumption_series(tl_dt)
    cons_right= get_consumption_series(tr_dt)

    sun_h = sunshine_hours(getattr(api_key, "LAT", 52.52),
                          getattr(api_key, "LON", 13.405))
    eco   = ecoflow_status()

    # Canvas
    img  = Image.new('1', (w, h), 255)
    d    = ImageDraw.Draw(img)

    # Fonts
    try:
        f_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        f_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        f_bold  = ImageFont.load_default()
        f_small = ImageFont.load_default()
        f_tiny  = ImageFont.load_default()
    fonts = {'bold': f_bold, 'small': f_small, 'tiny': f_tiny}

    # Top-Leiste: zwei Boxen
    margin = 10
    top_h  = 70
    box_w  = (w - margin*3)//2
    draw_weather_box(d, margin, margin, box_w, top_h, fonts, sun_h)
    draw_ecoflow_box(d, margin*2 + box_w, margin, box_w, top_h, fonts, eco)

    # Info-Zeile
    draw_info_box(d, info, fonts, y=top_h + margin + 10, width=w-20)

    # Chartbereich
    chart_top = top_h + margin + 30
    chart_area = (int(w*0.06), chart_top, w - int(w*0.06), h-30)

    draw_two_day_chart(
        d, left, right, fonts, labels, chart_area,
        pv_left=pv_left, pv_right=pv_right,
        cons_left=cons_left, cons_right=cons_right,
        cur_dt=info['current_dt'], cur_price=info['current_price']
    )

    # Footer
    footer = dt.datetime.now(LOCAL_TZ).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h-20), footer, font=fonts['tiny'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()
