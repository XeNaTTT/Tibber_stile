#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math, json, requests, datetime as dt, sqlite3, logging, random, time, hmac, hashlib
from urllib.parse import urlencode
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, numpy as np

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("Europe/Berlin")

# E-Paper Lib
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

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------- Tibber ----------
def tibber_priceinfo():
    if not getattr(api_key, "API_KEY", None) or str(api_key.API_KEY).startswith("DEIN_"):
        raise RuntimeError("Tibber API_KEY fehlt/Platzhalter. Trage einen gültigen Token in api_key.py ein.")
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    gql = (
        "{ viewer { homes { currentSubscription { priceInfo { "
        "today { total startsAt } "
        "tomorrow { total startsAt } "
        "current { total startsAt } "
        "}}}}}"
    )
    r = requests.post('https://api.tibber.com/v1-beta/gql',
                      json={"query": gql}, headers=hdr, timeout=20)
    if r.status_code >= 400:
        logging.error("Tibber HTTP %s: %s", r.status_code, r.text[:300])
        r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and j.get("errors"):
        raise RuntimeError(f"Tibber GraphQL Fehler: {j['errors']}")
    return j['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def prepare_info(pi):
    today = pi['today']
    today_vals = [s['total']*100 for s in today]
    cur = pi['current']
    low_idx = int(np.argmin(today_vals)) if today_vals else None
    low_time = (dt.datetime.fromisoformat(today[low_idx]['startsAt']).astimezone(LOCAL_TZ)
                if low_idx is not None else None)
    return {
        'current_dt': dt.datetime.fromisoformat(cur['startsAt']).astimezone(LOCAL_TZ),
        'current_price': cur['total']*100,
        'lowest_today': min(today_vals) if today_vals else 0,
        'lowest_today_time': low_time,
        'highest_today': max(today_vals) if today_vals else 0
    }

# ---------- 15-Min Transformation ----------
def expand_to_15min(slots):
    ts_list, val_list = [], []
    for s in slots:
        start = dt.datetime.fromisoformat(s['startsAt']).astimezone(LOCAL_TZ)
        price = s['total']*100
        for k in range(4):
            ts_list.append(start + dt.timedelta(minutes=15*k))
            val_list.append(price)
    return ts_list, val_list

# ---------- Tibber Consumption (hourly -> 15min) ----------
def tibber_hourly_consumption(last=48):
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type":"application/json"}
    q = f"""
    {{ viewer {{ homes {{
      consumption(resolution: HOURLY, last: {last}) {{
        nodes {{ from consumption }}
      }}
    }}}} }}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": q}, headers=hdr, timeout=15)
    r.raise_for_status()
    j = r.json()
    nodes = j["data"]["viewer"]["homes"][0]["consumption"]["nodes"]
    out = []
    for n in nodes:
        f = dt.datetime.fromisoformat(n["from"]).astimezone(LOCAL_TZ)
        out.append((f, float(n["consumption"] or 0.0)))
    return out

def upsample_hourly_to_quarter(ts_15min, hourly_list):
    import bisect
    if not hourly_list:
        return pd.Series([0.0]*len(ts_15min))
    hours = [t for (t,_) in hourly_list]
    vals  = [v for (_,v) in hourly_list]  # kWh je Stunde
    first_ts, last_ts = hours[0], hours[-1]
    out = []
    for t in ts_15min:
        if t < first_ts or t > last_ts:
            out.append(0.0)
            continue
        i = bisect.bisect_right(hours, t) - 1
        kwh = (vals[i] if i >= 0 else 0.0)
        out.append(kwh * 1000.0)  # konstante W pro 15-Min Slot
    return pd.Series(out)

# ---------- Wetter ----------
def sunshine_hours_both(lat, lon, model=None):
    """Gibt (heute_h, morgen_h) zurück. Open-Meteo sunshine_duration (Sek -> h)."""
    try:
        model = model or getattr(api_key, "SUN_MODEL", "icon_seamless")
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=sunshine_duration&timezone=Europe%2FBerlin"
            f"&models={model}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        try:
            with open("/home/alex/E-Paper-tibber-Preisanzeige/openmeteo_last.json", "w") as f:
                json.dump(j, f, indent=2)
        except Exception:
            pass
        arr = (j.get("daily", {}).get("sunshine_duration")) or []
        def _h(idx):
            try:
                sec = arr[idx]
                if sec is None: return 0.0
                return round(float(sec)/3600.0, 1)
            except Exception:
                return 0.0
        return _h(0), _h(1)
    except Exception as e:
        logging.error("Sunshine fetch failed: %s", e)
        return 0.0, 0.0

# ---------- EcoFlow (RAW) ----------
def _ef_sign(params: dict, secret_key: str) -> str:
    s = "&".join(f"{k}={v}" for k,v in sorted(params.items()))
    return hmac.new(secret_key.encode(), s.encode(), hashlib.sha256).hexdigest()

def ecoflow_status_raw():
    """
    Ruft die EcoFlow-API (quota/all) auf und gibt das *Roh*-Data-Dict zurück.
    Nutzt nur Roh-Keys wie powGetSysGrid, powGetSysLoad, powGetPvSum, cmsBattSoc, powGetBpCms ...
    """
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    sn   = getattr(api_key, "ECOFLOW_DEVICE_ID", "").strip()
    ak   = getattr(api_key, "ECOFLOW_APP_KEY", "").strip()
    sk   = getattr(api_key, "ECOFLOW_SECRET_KEY", "").strip()
    if not sn or not ak or not sk:
        raise RuntimeError("EcoFlow Keys/SN fehlen in api_key.py")

    path = "/iot-open/sign/device/quota/all"
    q = {"sn": sn}
    nonce = random.randint(100000, 999999)
    ts    = int(time.time() * 1000)
    sign_params = dict(q, **{"accessKey": ak, "nonce": nonce, "timestamp": ts})
    sig = _ef_sign(sign_params, sk)

    headers = {"accessKey": ak, "nonce": str(nonce), "timestamp": str(ts), "sign": sig}
    url = f"{base}{path}?{urlencode(q)}"
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()
    j = r.json()
    if str(j.get("code")) != "0":
        raise RuntimeError(f"EcoFlow quota/all Fehler: {j}")
    data = j.get("data", {}) or {}
    # Debug-Dump
    try:
        with open("/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_quota_last.json","w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    # Konsolen-Log (optional)
    print("----- EcoFlow API Response START -----")
    print(json.dumps(data, indent=2))
    print("----- EcoFlow API Response END -----")
    return data

# PV-Serie aus EcoFlow (konstant über Zeitraum)
def get_pv_series(slots_dt, eco):
    try:
        pv = eco.get("powGetPvSum", 0.0)
        pv_w = float(pv or 0.0)
        return pd.Series([pv_w] * len(slots_dt))
    except Exception as e:
        logging.error(f"PV aus EcoFlow fehlgeschlagen: {e}")
        return pd.Series([0.0] * len(slots_dt))

# ---------- Drawing ----------
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

def minutes_to_hhmm(m):
    if m is None: return "-"
    try:
        m = int(m)
        return f"{m//60:02d}:{m%60:02d} h"
    except: return "-"

def draw_battery(d, x, y, w, h, soc, arrow=None, fonts=None):
    try:
        soc = max(0, min(100, int(round(float(soc)))))
    except Exception:
        soc = 0
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    d.rectangle((x+w, y+h*0.35, x+w+6, y+h*0.65), outline=0, width=2)
    inner_w = max(0, int((w-6) * soc/100))
    d.rectangle((x+3, y+3, x+3+inner_w, y+h-3), fill=0)
    if fonts: d.text((x+w+12, y+h/2-7), f"{soc}%", font=fonts['small'], fill=0)
    if arrow == "up":
        d.polygon([(x+w+45,y+h*0.65),(x+w+55,y+h*0.65),(x+w+50,y*h*0.40)], fill=0)
    elif arrow == "down":
        d.polygon([(x+w+45,y*h*0.40),(x+w+55,y*h*0.40),(x+w+50,y*h*0.65)], fill=0)

def draw_weather_box(d, x, y, w, h, fonts, sun_today, sun_tomorrow=None):
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    # Sonne
    cx, cy, r = x+25, y+25, 10
    d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=0, width=2)
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        d.line((cx+math.cos(rad)*r*1.6, cy+math.sin(rad)*r*1.6,
                cx+math.cos(rad)*r*2.4, cy+math.sin(rad)*r*2.4), fill=0, width=2)
    d.text((x+60, y+5), "Wetter", font=fonts['bold'], fill=0)
    try: t_val = float(sun_today or 0.0)
    except: t_val = 0.0
    d.text((x+60, y+28), f"Sonnenstunden heute:  {t_val:.1f} h", font=fonts['small'], fill=0)
    if sun_tomorrow is not None:
        try: m_val = float(sun_tomorrow or 0.0)
        except: m_val = 0.0
        d.text((x+60, y+46), f"Sonnenstunden morgen: {m_val:.1f} h", font=fonts['small'], fill=0)

def draw_ecoflow_box_raw(d, x, y, w, h, fonts, eco):
    """
    Nutzt NUR Roh-Keys:
      SoC: cmsBattSoc (%)
      PV:  powGetPvSum (W)
      Netz: powGetSysGrid (W) oder gridConnectionPower (W)
      Last: powGetSysLoad (W)
      Leistung (Pfeil): powGetBpCms (negativ = Entladen) -> invertieren, sonst gridConnectionPower
    """
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    title_x, title_y = x+10, y+5
    d.text((title_x, title_y), "EcoFlow Stream AC", font=fonts['bold'], fill=0)

    def to_f(v):
        try: return float(v)
        except: return None
    def fmt_w(v):
        return f"{int(round(v))} W" if isinstance(v,(int,float)) else "-"

    soc   = eco.get("cmsBattSoc")
    pv    = to_f(eco.get("powGetPvSum"))
    grid  = to_f(eco.get("powGetSysGrid"))
    if grid is None:
        grid = to_f(eco.get("gridConnectionPower"))
    load  = to_f(eco.get("powGetSysLoad"))
    pwr   = to_f(eco.get("powGetBpCms"))
    if pwr is not None:
        pwr = -pwr  # >0 Entladen, <0 Laden (Vorzeichen drehen)
    else:
        pwr = to_f(eco.get("gridConnectionPower"))

    # Pfeil neben Titel (Pfeil 15 px weiter rechts)
    arrow = None
    if isinstance(pwr,(int,float)):
        if pwr < -10: arrow = "up"
        elif pwr > 10: arrow = "down"
    arrow_offset = 15
    if arrow == "up":
        d.polygon([(title_x + 150 + arrow_offset, title_y + 20),
                   (title_x + 160 + arrow_offset, title_y + 20),
                   (title_x + 155 + arrow_offset, title_y + 5)], fill=0)
    elif arrow == "down":
        d.polygon([(title_x + 150 + arrow_offset, title_y + 5),
                   (title_x + 160 + arrow_offset, title_y + 5),
                   (title_x + 155 + arrow_offset, title_y + 20)], fill=0)

    # Batterie (SoC)
    batt_x, batt_y = x+10, y+28
    draw_battery(d, batt_x, batt_y, 90, 28, soc, arrow=None, fonts=fonts)

    # Zwei Spalten, Texte 10 px weiter rechts
    text_offset = 10
    left_x  = batt_x + 120 + text_offset
    right_x = batt_x + 260 + text_offset
    base_y  = batt_y - 4
    line_h  = 16

    d.text((left_x,  base_y + 0*line_h), f"Leistung: {fmt_w(pwr)}", font=fonts['small'], fill=0)
    d.text((left_x,  base_y + 1*line_h), f"PV-Ertrag: {fmt_w(pv)}",  font=fonts['small'], fill=0)
    d.text((right_x, base_y + 0*line_h), f"Netz: {fmt_w(grid)}",    font=fonts['small'], fill=0)
    d.text((right_x, base_y + 1*line_h), f"Last: {fmt_w(load)}",    font=fonts['small'], fill=0)

def draw_info_box(d, info, fonts, y, width):
    x0 = 10
    low_time = info.get('lowest_today_time')
    low_lbl = (f"{info['lowest_today']/100:.2f} ct @ {low_time.strftime('%H:%M')}"
               if low_time else f"{info['lowest_today']/100:.2f} ct")
    items = [
        ("Preis jetzt", f"{info['current_price']/100:.2f} ct"),
        ("Tief heute",  low_lbl),
        ("Hoch heute",  f"{info['highest_today']/100:.2f} ct"),
    ]
    colw = width/len(items)
    for i,(k,v) in enumerate(items):
        d.text((x0 + i*colw, y), f"{k}: {v}", font=fonts['bold'], fill=0)

def draw_two_day_chart(d, left, right, fonts, subtitles, area,
                       pv_left=None, pv_right=None,
                       cons_left=None, cons_right=None,
                       cur_dt=None, cur_price=None):
    X0,Y0,X1,Y1 = area
    W,H = X1-X0, Y1-Y0
    PW  = W/2

    tl, vl = expand_to_15min(left)
    tr, vr = expand_to_15min(right)
    if not (vl or vr): return

    allp = vl + vr
    vmin, vmax = min(allp)-0.5, max(allp)+0.5
    sy_price = H/(vmax - vmin if vmax>vmin else 1)

    def vmax_power(series):
        if series is None: return 0
        try: return float(np.nanmax(series)) if len(series)>0 else 0
        except: return 0
    pmax = max(vmax_power(pv_left), vmax_power(pv_right),
               vmax_power(cons_left), vmax_power(cons_right))
    sy_pow = H/((pmax or 1)*1.5)  # etwas „tiefer“ skalieren

    # Preis-Y Ticks (nur innerhalb des Chart-Bereichs)
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        yy = Y1 - (yv - vmin) * sy_price
        if Y0 < yy < Y1:
            d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['tiny'], fill=0)
        yv += step

    def panel(ts_list, val_list, pv_list, cons_list, x0):
        n = len(ts_list)
        if n < 2: return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        # Preis-Stufenlinie
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (val_list[i]   - vmin)*sy_price
            x2, y2 = xs[i+1], Y1 - (val_list[i+1] - vmin)*sy_price
            d.line((x1,y1, x2,y1), fill=0, width=2)
            d.line((x2,y1, x2,y2), fill=0, width=2)
        # PV (gestrichelt)
        if pv_list is not None and n == len(pv_list):
            for i in range(n-1):
                y1 = Y1 - pv_list.iloc[i]*sy_pow
                y2 = Y1 - pv_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=2, gap=2, width=1)
        # Verbrauch (gestrichelt, längere Striche)
        if cons_list is not None and n == len(cons_list):
            for i in range(n-1):
                y1 = Y1 - cons_list.iloc[i]*sy_pow
                y2 = Y1 - cons_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=4, gap=3, width=1)
        # Min/Max-Labels Preis
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], Y1 - (val_list[idx]-vmin)*sy_price
            d.text((xi-12, yi-12), f"{val_list[idx]/100:.2f}", font=fonts['tiny'], fill=0)

    panel(tl, vl, pv_left,  cons_left,  X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(tr, vr, pv_right, cons_right, X0+PW)

    # Subtitles
    d.text((X0+5,    Y1+28), subtitles[0], font=fonts['bold'], fill=0)
    d.text((X0+PW+5, Y1+28), subtitles[1], font=fonts['bold'], fill=0)

    # Stundenbeschriftung
    def hour_ticks(ts_list, x0):
        if len(ts_list) < 2: return
        n = len(ts_list)
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        for i,t in enumerate(ts_list):
            if t.minute == 0:
                d.line((xs[i], Y1, xs[i], Y1+4), fill=0, width=1)
                d.text((xs[i]-8, Y1+6), t.strftime("%H"), font=fonts['tiny'], fill=0)
    hour_ticks(tl, X0)
    hour_ticks(tr, X0+PW)

    # Legende (ASCII-Striche gegen Encoding-Probleme)
    d.text((X1-180, Y0-16), "--  PV   ----  Verbrauch", font=fonts['tiny'], fill=0)

    # Minutengenauer Marker
    if cur_price is not None:
        now = dt.datetime.now(LOCAL_TZ)
        def pick_panel_for_now():
            if len(tl) > 1 and tl[0] <= now <= tl[-1]:
                return tl, X0
            if len(tr) > 1 and tr[0] <= now <= tr[-1]:
                return tr, X0 + PW
            return None, None
        arr, x0 = pick_panel_for_now()
        if arr is not None:
            n = len(arr)
            if n > 1:
                t0 = arr[0]
                i_float = (now - t0).total_seconds() / 900.0
                i_float = max(0.0, min(n - 1, i_float))
                slot_w = PW / (n - 1)
                px = x0 + i_float * slot_w
                py = Y1 - (cur_price - vmin) * sy_price
                r = 4
                d.ellipse((px - r, py - r, px + r, py + r), fill=0)
                d.text((px + r + 2, py - r - 2), f"{cur_price/100:.2f}", font=fonts['tiny'], fill=0)

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height

    # Daten laden (Tibber)
    try:
        pi = tibber_priceinfo()
        update_price_cache(pi)
    except Exception as e:
        logging.error("Nutze Cache wegen Fehler: %s", e)
        today_cache = load_cache(CACHE_TODAY) or {"data": []}
        pi = {
            'today': today_cache.get('data', []),
            'tomorrow': [],
            'current': {'startsAt': dt.datetime.now(LOCAL_TZ).isoformat(),
                        'total': (today_cache.get('data',[{'total':0}])[0].get('total', 0) or 0)}
        }

    info = prepare_info(pi)
    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left, right = pi['today'], tomorrow
        labels = ("Heute", "Morgen")
    else:
        left, right = (load_cache(CACHE_YESTERDAY) or {"data": []})['data'], pi['today']
        labels = ("Gestern", "Heute")

    tl_dt, _ = expand_to_15min(left)
    tr_dt, _ = expand_to_15min(right)

    # EcoFlow Rohdaten
    eco = {}
    try:
        eco = ecoflow_status_raw()
    except Exception as e:
        logging.error(f"EcoFlow Status fehlgeschlagen: {e}")
        eco = {}

    # PV (aus EcoFlow)
    pv_left   = get_pv_series(tl_dt, eco=eco)
    pv_right  = get_pv_series(tr_dt, eco=eco)

    # Tibber-Verbrauch (hochgerechnet auf 15 min)
    hourly = []
    try:
        hourly = tibber_hourly_consumption(last=48)
    except Exception as e:
        logging.error(f"Tibber Consumption fehlgeschlagen: {e}")
    cons_left  = upsample_hourly_to_quarter(tl_dt, hourly)
    cons_right = upsample_hourly_to_quarter(tr_dt, hourly)

    sun_today, sun_tomorrow = sunshine_hours_both(api_key.LAT, api_key.LON)

    # Canvas
    img  = Image.new('1', (w, h), 255)
    d    = ImageDraw.Draw(img)

    # Fonts
    try:
        f_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        f_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        f_bold = f_small = f_tiny = ImageFont.load_default()
    fonts = {'bold': f_bold, 'small': f_small, 'tiny': f_tiny}

    # Layout
    margin = 10
    top_h  = 70
    box_w  = (w - margin*3)//2
    draw_weather_box(d, margin, margin, box_w, top_h, fonts, sun_today, sun_tomorrow)
    draw_ecoflow_box_raw(d, margin*2 + box_w, margin, box_w, top_h, fonts, eco)

    # Info-Zeile tiefer
    draw_info_box(d, info, fonts, y=top_h + margin + 6, width=w-20)

    # Chart kleiner + Platz für Stunden
    chart_top = top_h + margin + 40
    chart_area = (int(w*0.06), chart_top, w - int(w*0.06), h-70)

    draw_two_day_chart(
        d, left, right, fonts, labels, chart_area,
        pv_left=pv_left, pv_right=pv_right,
        cons_left=cons_left, cons_right=cons_right,
        cur_dt=info['current_dt'], cur_price=info['current_price']
    )

    footer = dt.datetime.now(LOCAL_TZ).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h-10), footer, font=fonts['tiny'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__ == "__main__":
    main()