#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math, json, requests, datetime as dt, sqlite3, logging
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, numpy as np
from urllib.parse import urlencode
import re

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
def _to_float(x):
    """Robuste Zahl-Konvertierung: akzeptiert int/float/Strings (inkl. Vorzeichen, Komma)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    # erlaube +/-, eine Dezimalstelle, keine anderen Zeichen
    m = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s)
    return float(s) if m else None

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
    if not getattr(api_key, "API_KEY", None) or str(api_key.API_KEY).startswith("DEIN_"):
        raise RuntimeError("Tibber API_KEY fehlt/Platzhalter. Trage einen gÃ¼ltigen Token in api_key.py ein.")
    hdr = {
        "Authorization": f"Bearer {api_key.API_KEY}",
        "Content-Type": "application/json"
    }
    gql = (
        "{ viewer { homes { currentSubscription { priceInfo { "
        "today { total startsAt } "
        "tomorrow { total startsAt } "
        "current { total startsAt } "
        "}}}}}"
    )
    try:
        r = requests.post('https://api.tibber.com/v1-beta/gql',
                          json={"query": gql}, headers=hdr, timeout=20)
        if r.status_code >= 400:
            logging.error("Tibber HTTP %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        j = r.json()
    except Exception as e:
        raise RuntimeError(f"Tibber Request fehlgeschlagen: {e}")
    if isinstance(j, dict) and j.get("errors"):
        raise RuntimeError(f"Tibber GraphQL Fehler: {j['errors']}")
    try:
        return j['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    except Exception as e:
        raise RuntimeError(f"Tibber Antwort unerwartet: {e}, payload keys: {list(j.keys())}")

def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}

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

# ---------- DB-Serien ----------
def series_from_db(table, column, slots_dt):
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(f"SELECT ts, {column} FROM {table}", conn)
    except Exception:
        conn.close(); return pd.Series([0.0]*len(slots_dt))
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index('ts', inplace=True)
    df = df.resample('15T').mean().ffill().fillna(0)
    out = []
    for t in slots_dt:
        v = df[column].asof(t) if not df.empty else 0.0
        out.append(float(0.0 if pd.isna(v) else v))
    return pd.Series(out)

def get_pv_series(slots_dt):
    return series_from_db("pv_log", "dtu_power", slots_dt)

def get_consumption_series(slots_dt):
    return series_from_db("consumption_log", "consumption_w", slots_dt)

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
        out.append(kwh * 1000.0)  # W pro 15-Minuten-Slot (vereinfachtes Profil)
    return pd.Series(out)

# ---------- Wetter ----------
def sunshine_hours_both(lat, lon, model=None):
    """
    Liefert (heute_h, morgen_h) als floats (h).
    Nutzt Open-Meteo sunshine_duration (Sekunden -> Stunden).
    """
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
                if sec is None:
                    return 0.0
                return round(float(sec)/3600.0, 1)
            except Exception:
                return 0.0
        return _h(0), _h(1)
    except Exception as e:
        logging.error("Sunshine fetch failed: %s", e)
        return 0.0, 0.0

# ---------- EcoFlow (BKW/PowerStream, signierte Requests) ----------
import time, uuid, hmac, hashlib
from urllib.parse import urlencode

def _six_digit_nonce():
    # 6-stelliger Nonce, wie in der Doku gefordert
    return f"{int(time.time()*1000) % 900000 + 100000}"

def _flatten_params(obj, prefix=""):
    """
    Flacht dict/list gemÃ¤ÃŸ Doku ab:
    - dict:   deviceInfo.id=1
    - list:   ids[0]=1&ids[1]=2
    - nested: params.cmdSet=11&params.id=24 ...
    Liefert Liste (key, value) -> spÃ¤ter ASCII-sortiert.
    """
    items = []
    if isinstance(obj, dict):
        for k in obj:
            key = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_params(obj[k], key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            items.extend(_flatten_params(v, key))
    else:
        items.append((prefix, "" if obj is None else str(obj)))
    return items

def _build_sign_string(params_dict, access_key, nonce, timestamp):
    """
    1) Body/Query-Objekt flatten + ASCII-sortieren
    2) accessKey, nonce, timestamp anhÃ¤ngen
    3) Ergebnis-String fÃ¼r HMAC (UTF-8)
    """
    kv = _flatten_params(params_dict) if params_dict else []
    kv.sort(key=lambda kv_: kv_[0])  # ASCII-sortiert
    base = "&".join(f"{k}={v}" for k, v in kv) if kv else ""
    tail = f"accessKey={access_key}&nonce={nonce}&timestamp={timestamp}"
    return (base + "&" + tail) if base else tail

def _hmac_sha256_hex(secret_key, msg):
    return hmac.new(secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _signed_headers(access_key, secret_key, params_dict, content_type=None):
    ts = str(int(time.time()*1000))
    nonce = _six_digit_nonce()
    sign_str = _build_sign_string(params_dict, access_key, nonce, ts)
    sig = _hmac_sha256_hex(secret_key, sign_str)
    hdr = {
        "accessKey": access_key,
        "nonce": nonce,
        "timestamp": ts,
        "sign": sig
    }
    if content_type:
        hdr["Content-Type"] = content_type
    return hdr

def ecoflow_get_device_list():
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/list"
    params = {}  # keine GET-Parameter
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, params, content_type=None)
    url = f"{base}{path}"
    r = requests.get(url, headers=hdr, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}
    if r.status_code == 200 and str(j.get("code")) == "0":
        return j.get("data", []) or []
    raise RuntimeError(f"EcoFlow device/list fehlgeschlagen: HTTP {r.status_code}, resp={str(j)[:200]}")

def ecoflow_get_all_quota(sn):
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/quota/all"
    query = {"sn": sn}
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, query, content_type=None)
    url = f"{base}{path}?{urlencode(query)}"
    r = requests.get(url, headers=hdr, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}
    if r.status_code == 200 and str(j.get("code")) == "0":
        return j.get("data", {}) or {}
    raise RuntimeError(f"EcoFlow quota/all fehlgeschlagen: HTTP {r.status_code}, resp={str(j)[:200]}")

def ecoflow_status_bkw():
    """
    Stream/PowerStream-Mapping:
      - soc:           cmsBattSoc (%)
      - pv_input_w:    powGetPvSum (W)
      - load_w:        powGetSysLoad (W)
      - grid_w:        powGetSysGrid (W) oder gridConnectionPower (W)
      - power_w:       abgeleitet = load_w - pv_w - grid_w
                       (>0 = Entladen, <0 = Laden)
      - eta_min:       (für Stream oft nicht vorhanden -> None)
      - mode:          aus feedGridMode / energyStrategyOperateMode.*
    Schreibt Rohdaten nach ecoflow_quota_last.json.
    """
    sn = getattr(api_key, "ECOFLOW_DEVICE_ID", "").strip()
    if not sn:
        raise RuntimeError("ECOFLOW_DEVICE_ID fehlt in api_key.py")

    try:
        q = ecoflow_get_all_quota(sn)  # dict mit flachen Keys
        # Dump zum Nachsehen
        try:
            with open("/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_quota_last.json", "w") as f:
                json.dump(q, f, indent=2)
        except Exception:
            pass
    except Exception as e:
        logging.error("EcoFlow quota/all fehlgeschlagen: %s", e)
        if os.path.exists(ECOFLOW_FALLBACK):
            try:
                with open(ECOFLOW_FALLBACK) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "soc": None, "power_w": None, "mode": None, "eta_min": None,
            "pv_input_w_sum": None, "pv1_input_w": None, "pv2_input_w": None
        }

    # --- Helper zum Auslesen ---
    def get_num(key, default=None):
        v = q.get(key)
        fv = _to_float(v)
        return fv if fv is not None else default

    # ---- SoC ----
    soc = get_num("cmsBattSoc")
    if soc is None:
        soc = get_num("cmsMinDsgSoc")  # schlechter Fallback
    if soc is not None:
        try: soc = int(round(soc))
        except: pass

    # ---- PV ----
    pv_sum = get_num("powGetPvSum")

    # ---- Load & Grid ----
    load_w = get_num("powGetSysLoad")
    grid_w = get_num("powGetSysGrid")
    if grid_w is None:
        grid_w = get_num("gridConnectionPower")

    # ---- Batterie-Leistung aus Bilanz ----
    power_w = None
    if (load_w is not None) and (pv_sum is not None) and (grid_w is not None):
        # Konvention: >0 = Entladen (Abgabe), <0 = Laden (Aufnahme)
        power_w = (load_w - pv_sum - grid_w)
    else:
        # Wenn einzelne Teile fehlen, wenigstens etwas Sinnvolles anzeigen:
        # Falls nur grid & pv fehlen, nimm load als Proxy (not ideal).
        if load_w is not None and pv_sum is not None:
            power_w = (load_w - pv_sum)
        elif load_w is not None and grid_w is not None:
            power_w = (load_w - grid_w)
        elif pv_sum is not None and grid_w is not None:
            power_w = -(pv_sum + grid_w)  # sehr grob

    # ---- Restzeit (Stream hat meist keine) ----
    eta_min = None  # bleibt i.d.R. None bei Stream/PowerStream

    # ---- Modus / Flags ----
    # feedGridMode: 0/1 → Einspeisen erlaubt?
    feed_mode = q.get("feedGridMode")
    es_self   = q.get("energyStrategyOperateMode.operateSelfPoweredOpen")
    es_sched  = q.get("energyStrategyOperateMode.operateIntelligentScheduleModeOpen")
    mode = None
    if isinstance(feed_mode, (int,float,str)):
        try:
            mode = f"FeedMode:{int(float(feed_mode))}"
        except:
            mode = f"FeedMode:{feed_mode}"
    if es_self not in (None, ""):
        mode = (mode + " | " if mode else "") + f"Self:{es_self}"
    if es_sched not in (None, ""):
        mode = (mode + " | " if mode else "") + f"Sched:{es_sched}"

    return {
        "soc": soc,
        "power_w": power_w,
        "mode": mode,
        "eta_min": eta_min,
        "pv_input_w_sum": pv_sum,
        "pv1_input_w": None,
        "pv2_input_w": None
    }



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

# ---------- Helper ----------
def _as_float_or_none(x):
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def draw_weather_box(d, x, y, w, h, fonts, sun_today, sun_tomorrow=None):
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    cx, cy, r = x+25, y+25, 10
    d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=0, width=2)
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        d.line((cx+math.cos(rad)*r*1.6, cy+math.sin(rad)*r*1.6,
                cx+math.cos(rad)*r*2.4, cy+math.sin(rad)*r*2.4), fill=0, width=2)
    d.text((x+60, y+5), "Wetter", font=fonts['bold'], fill=0)
    try:  t_val = float(sun_today)    if sun_today    is not None else 0.0
    except: t_val = 0.0
    try:  m_val = float(sun_tomorrow) if sun_tomorrow is not None else None
    except: m_val = None
    d.text((x+60, y+28), f"Sonnenstunden heute:  {t_val:.1f} h", font=fonts['small'], fill=0)
    if m_val is not None:
        d.text((x+60, y+46), f"Sonnenstunden morgen: {m_val:.1f} h", font=fonts['small'], fill=0)

def minutes_to_hhmm(m):
    if m is None: return "â€”"
    try:
        m = int(m); return f"{m//60:02d}:{m%60:02d} h"
    except: return "â€”"

def draw_battery(d, x, y, w, h, soc, arrow=None, fonts=None):
    soc = max(0, min(100, int(soc) if soc is not None else 0))
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    d.rectangle((x+w, y+h*0.35, x+w+6, y+h*0.65), outline=0, width=2)
    inner_w = max(0, int((w-6) * soc/100))
    d.rectangle((x+3, y+3, x+3+inner_w, y+h-3), fill=0)
    if fonts: d.text((x+w+12, y+h/2-7), f"{soc}%", font=fonts['small'], fill=0)
    if arrow == "up":
        d.polygon([(x+w+45,y+h*0.65),(x+w+55,y+h*0.65),(x+w+50,y+h*0.40)], fill=0)
    elif arrow == "down":
        d.polygon([(x+w+45,y+h*0.40),(x+w+55,y+h*0.40),(x+w+50,y+h*0.65)], fill=0)

def draw_ecoflow_box(d, x, y, w, h, fonts, st):
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    d.text((x+10, y+5), "EcoFlow Stream AC", font=fonts['bold'], fill=0)
    arrow=None
    mode = (str(st.get('mode')) or "").lower()
    p = st.get('power_w')
    if "charg" in mode: arrow="up"
    elif "discharg" in mode or "feed" in mode: arrow="down"
    elif isinstance(p,(int,float)):
        if p < -10: arrow="up"
        elif p > 10: arrow="down"
    batt_x, batt_y = x+10, y+28
    draw_battery(d, batt_x, batt_y, 90, 28, st.get('soc'), arrow=arrow, fonts=fonts)
    lines = [
        f"Leistung: {int(p)} W" if isinstance(p,(int,float)) else "Leistung: â€”",
        f"Modus: {st.get('mode') or 'â€”'}",
        f"Restzeit: {minutes_to_hhmm(st.get('eta_min'))}"
    ]
    # optional PV-Zeile, wenn vorhanden
    pv_sum = st.get('pv_input_w_sum')
    if isinstance(pv_sum,(int,float)):
        lines.insert(0, f"PV ges.: {int(pv_sum)} W")
    for i, t in enumerate(lines):
        d.text((batt_x+120, batt_y - 4 + i*16), t, font=fonts['small'], fill=0)

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
    sy_pow = H/((pmax or 1)*1.2)

    # Preis-Y-Ticks (nur innerhalb)
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
        # Preis Stufenlinie
        for i in range(n-1):
            x1, y1 = xs[i],   Y1 - (val_list[i]   - vmin)*sy_price
            x2, y2 = xs[i+1], Y1 - (val_list[i+1] - vmin)*sy_price
            d.line((x1,y1, x2,y1), fill=0, width=2)
            d.line((x2,y1, x2,y2), fill=0, width=2)
        # PV
        if pv_list is not None and n == len(pv_list):
            for i in range(n-1):
                y1 = Y1 - pv_list.iloc[i]*sy_pow
                y2 = Y1 - pv_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=2, gap=2, width=1)
        # Verbrauch
        if cons_list is not None and n == len(cons_list):
            for i in range(n-1):
                y1 = Y1 - cons_list.iloc[i]*sy_pow
                y2 = Y1 - cons_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=4, gap=3, width=1)
        # Min/Max Labels
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], Y1 - (val_list[idx]-vmin)*sy_price
            d.text((xi-12, yi-12), f"{val_list[idx]/100:.2f}", font=fonts['tiny'], fill=0)

    panel(tl, vl, pv_left,  cons_left,  X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(tr, vr, pv_right, cons_right, X0+PW)

    # Subtitles unter Achse
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

    # Legende Leistung
    d.text((X1-180, Y0-16), "â€” â€” PV   â€”â€”  Verbrauch", font=fonts['tiny'], fill=0)

    # Minutengenauer Marker (horizontale Interpolation)
    if cur_price is not None:
        now = dt.datetime.now(LOCAL_TZ)

        def pick_panel_for_now():
            if len(tl) > 1 and tl[0] <= now <= tl[-1]:
                return tl, X0
            if len(tr) > 1 and tr[0] <= now <= tr[-1]:
                return tr, X0 + PW
            return None, None

        arr, x0_panel = pick_panel_for_now()
        if arr is not None:
            n = len(arr)
            if n > 1:
                t0 = arr[0]
                i_float = (now - t0).total_seconds() / 900.0  # 900s = 15 min
                i_float = max(0.0, min(n - 1, i_float))
                slot_w = PW / (n - 1)
                px = x0_panel + i_float * slot_w
                py = Y1 - (cur_price - vmin) * sy_price
                r = 4
                d.ellipse((px - r, py - r, px + r, py + r), fill=0)
                d.text((px + r + 2, py - r - 2), f"{cur_price/100:.2f}", font=fonts['tiny'], fill=0)

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height

    # Daten laden, robust gegen API-Ausfall
    try:
        pi = tibber_priceinfo()
        update_price_cache(pi)
    except Exception as e:
        logging.error("Nutze Cache wegen Fehler: %s", e)
        today_cache = load_cache(CACHE_TODAY) or {"data": []}
        yesterday_cache = load_cache(CACHE_YESTERDAY) or {"data": []}
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

    pv_left   = get_pv_series(tl_dt)
    pv_right  = get_pv_series(tr_dt)

    hourly = tibber_hourly_consumption(last=48)
    cons_left  = upsample_hourly_to_quarter(tl_dt, hourly)
    cons_right = upsample_hourly_to_quarter(tr_dt, hourly)

    sun_today, sun_tomorrow = sunshine_hours_both(api_key.LAT, api_key.LON)

    # EcoFlow: signierte BKW-Variante
    eco = ecoflow_status_bkw()

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
    draw_ecoflow_box(d, margin*2 + box_w, margin, box_w, top_h, fonts, eco)

    # Info-Zeile tiefer
    draw_info_box(d, info, fonts, y=top_h + margin + 6, width=w-20)

    # Chart kleiner in der HÃ¶he + Platz fÃ¼r Stunden
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

# Einmal ausführen:
data = ecoflow_get_all_quota(api_key.ECOFLOW_DEVICE_ID)
print("EcoFlow keys:", sorted(list(data.keys()))[:40], "...")  # erste ~40 Keys

if __name__ == "__main__":
    main()
