#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math, json, requests, datetime as dt, sqlite3, logging
from PIL import Image, ImageDraw, ImageFont, ImageChops
import pandas as pd, numpy as np
from urllib.parse import urlencode
import re

ECO_DEBUG = bool(int(os.getenv("ECO_DEBUG", "0")))
PV_PAT = re.compile(r"(pv|solar|yield|gen|power|input|watt|energy)", re.I)
DUMP_DIR = "/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_dump"
# Solar-Historie laut EcoFlow-Doku (PV-Linie)
ECOFLOW_SOLAR_ENERGY_CODE = "BK621_SOLAR-ENERGY||||"
ECOFLOW_PV1_CODE = None
ECOFLOW_PV2_CODE = None
ECOFLOW_PV_TOTAL_CODE = None

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
TIBBER_LAST_QUARTER_RESPONSE = '/home/alex/E-Paper-tibber-Preisanzeige/tibber_last_quarter_response.json'

logging.basicConfig(level=logging.INFO)
SUN_TODAY = None
SUN_TOMORROW = None

# Rendering pipeline
RENDER_SCALE = 2
DITHER_MODE = "BAYER"  # "BAYER" or "FS"
BAYER_MATRIX = 8       # 8 or 16
GAMMA = 1.10
CONTRAST = 1.08

# Weather icon config
WEATHER_ICON_DIR = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/Wettersymbole"
WEATHER_ICON_WIDTH = 240
WEATHER_ICON_HEIGHT = 235
ICON_INVERT = True
ICON_BITREVERSE = False
_C_BITMAP_CACHE = {}
_C_IMAGE_CACHE = {}
_BIT_REVERSE_TABLE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))

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


def _dump_json(name, obj):
    if not ECO_DEBUG:
        return None
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn = os.path.join(DUMP_DIR, f"{name}_{ts}.json")
        with open(fn, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return fn
    except Exception as e:
        try:
            logging.debug("EcoFlow debug dump skipped: %s", e)
        except Exception:
            pass
    return None

def _dump_json_force(name, obj):
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn = os.path.join(DUMP_DIR, f"{name}_{ts}.json")
        with open(fn, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return fn
    except Exception as e:
        try:
            logging.debug("EcoFlow debug dump(force) skipped: %s", e)
        except Exception:
            pass
    return None

def _pv_candidates(d):
    if not isinstance(d, dict):
        return []
    out = []
    for k, v in d.items():
        try:
            if not PV_PAT.search(str(k)):
                continue
            example = v
            if isinstance(v, (list, tuple)) and v:
                example = v[0]
            elif isinstance(v, dict) and v:
                example = next(iter(v.values()))
            out.append((k, type(v).__name__, example))
        except Exception:
            continue
    return out


def pick(src, keys):
    res = {}
    if not isinstance(src, dict):
        return res
    for k in keys:
        try:
            if k in src:
                res[k] = src.get(k)
        except Exception:
            continue
    return res

def load_c_bitmap(path, varname):
    if path in _C_BITMAP_CACHE:
        return _C_BITMAP_CACHE[path]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Icon-Datei nicht lesbar: {path}: {e}")
    if varname not in content:
        raise RuntimeError(f"Array {varname} nicht gefunden in {path}")
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    content = re.sub(r"//.*", "", content)
    bytes_list = [int(b, 16) for b in re.findall(r"0[xX][0-9A-Fa-f]{2}", content)]
    if not bytes_list:
        raise RuntimeError(f"Keine Icon-Daten gefunden in {path}")
    data = bytes(bytes_list)
    _C_BITMAP_CACHE[path] = data
    return data

def c_bitmap_to_image(data, w, h, invert=False, bitreverse=False):
    if bitreverse:
        data = data.translate(_BIT_REVERSE_TABLE)
    img = Image.frombytes("1", (w, h), data)
    if invert:
        img = ImageChops.invert(img)
    return img

def get_weather_icon_from_bucket(bucket, is_day):
    bucket = bucket or "cloudy"
    if bucket == "clear":
        return "sun.c"
    if bucket in ("partly", "cloudy", "overcast", "fog"):
        return "cloudy.c"
    if bucket in ("rain", "drizzle", "thunder"):
        return "rain.c"
    if bucket == "snow":
        return "cloudy.c"
    return "cloudy.c"

def _get_weather_icon_image(bucket, is_day, invert=False, bitreverse=False):
    filename = get_weather_icon_from_bucket(bucket, is_day)
    if not filename:
        return None
    path = os.path.join(WEATHER_ICON_DIR, filename)
    varname = f"gImage_{os.path.splitext(filename)[0]}"
    data = load_c_bitmap(path, varname)
    w = WEATHER_ICON_WIDTH
    bytes_per_row = (w + 7) // 8
    if len(data) % bytes_per_row != 0:
        raise RuntimeError(
            "Ungültige Icon-Datenlänge: len(data)=%d, bytes_per_row=%d"
            % (len(data), bytes_per_row)
        )
    h = len(data) // bytes_per_row
    if h < 50 or h > 400:
        logging.warning(
            "Auffällige Icon-Höhe berechnet: %d (len(data)=%d, bytes_per_row=%d)",
            h,
            len(data),
            bytes_per_row,
        )
    cache_key = (path, w, h, invert, bitreverse)
    if cache_key in _C_IMAGE_CACHE:
        return _C_IMAGE_CACHE[cache_key]
    img = c_bitmap_to_image(data, w, h, invert=invert, bitreverse=bitreverse)
    _C_IMAGE_CACHE[cache_key] = img
    return img

# ---------- Tibber ----------
def pick_home_with_data(homes):
    """Wählt das erste Home, das verwertbare Daten hat (PriceInfo oder Consumption)."""
    if not homes:
        return None

    # Prefer: Home mit currentSubscription.priceInfo.today
    for h in homes:
        cs = (h or {}).get("currentSubscription") or {}
        pi = (cs.get("priceInfo") or {})
        if pi.get("today"):
            return h

    # Fallback: Home mit consumption.nodes
    for h in homes:
        cons = (h or {}).get("consumption") or {}
        nodes = cons.get("nodes") if isinstance(cons, dict) else None
        if nodes:
            return h

    return homes[0]

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
        data = (j or {}).get("data") or {}
        viewer = data.get("viewer") or {}
        homes = viewer.get("homes") or []
        home = pick_home_with_data(homes) or {}
        cs = (home.get("currentSubscription") or {})
        pi = (cs.get("priceInfo") or {})
        logging.info(
            "Tibber: homes=%d, picked_home_has_priceinfo=%s",
            len(homes),
            bool(((home.get("currentSubscription") or {}).get("priceInfo") or {}).get("today"))
        )
        if not pi or not pi.get("today"):
            raise RuntimeError(f"Tibber Antwort unerwartet/leer: data_keys={list(data.keys())}")
        logging.info(
            "Tibber Preisinfo via API: heute=%d, morgen=%d, current=%s",
            len(pi.get("today", []) or []),
            len(pi.get("tomorrow", []) or []),
            safe_get(pi, "current", "startsAt", default="-")
        )
        return pi
    except Exception as e:
        raise RuntimeError(f"Tibber Antwort unerwartet: {e}, payload keys: {list((j or {}).keys())}")


def tibber_priceinfo_quarter_range():
    """
    Versucht 15-Minuten-Preise über GraphQL (resolution: QUARTER_HOURLY) zu laden.
    Liefert None zurück, wenn die API dies nicht unterstützt oder keine Daten liefert.
    """
    if not getattr(api_key, "API_KEY", None) or str(api_key.API_KEY).startswith("DEIN_"):
        logging.error("Tibber API_KEY fehlt/Platzhalter. Trage einen gültigen Token in api_key.py ein.")
        return None

    hdr = {
        "Authorization": f"Bearer {api_key.API_KEY}",
        "Content-Type": "application/json"
    }
    gql = (
        "{ viewer { homes { id appNickname currentSubscription { "
        "priceInfo(resolution: QUARTER_HOURLY) { "
        "today { total startsAt } "
        "tomorrow { total startsAt } "
        "current { total startsAt } "
        "} } } } }"
    )

    r = None
    try:
        r = requests.post(
            'https://api.tibber.com/v1-beta/gql',
            json={"query": gql},
            headers=hdr,
            timeout=20
        )
        http_status = r.status_code
        try:
            resp_json = r.json()
        except Exception:
            resp_json = {"raw": r.text}
        try:
            with open(TIBBER_LAST_QUARTER_RESPONSE, "w") as f:
                json.dump(
                    {
                        "ts": dt.datetime.now(LOCAL_TZ).isoformat(),
                        "http_status": http_status,
                        "response": resp_json
                    },
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            logging.debug("Konnte Tibber-Quarter-Response nicht schreiben: %s", e)

        if http_status >= 400:
            logging.error("Tibber HTTP %s: %s", http_status, r.text[:300])
            return None

        if isinstance(resp_json, dict) and resp_json.get("errors"):
            logging.error("Tibber GraphQL Fehler: %s", resp_json.get("errors"))
            return None

        data = (resp_json or {}).get("data") or {}
        viewer = data.get("viewer") or {}
        homes = viewer.get("homes") or []
        if not homes:
            logging.error("Tibber Quarter: keine Homes in der Antwort")
            return None

        home = None
        for h in homes:
            cs = (h or {}).get("currentSubscription") or {}
            pi = (cs.get("priceInfo") or {})
            if pi.get("today"):
                home = h
                break
        if not home:
            logging.error("Tibber Quarter: kein Home mit priceInfo.today gefunden")
            return None

        pi = ((home.get("currentSubscription") or {}).get("priceInfo") or {})
        return {
            "source": "tibber_quarter",
            "home_id": home.get("id"),
            "today": pi.get("today") or [],
            "tomorrow": pi.get("tomorrow") or [],
            "current": pi.get("current") or {}
        }
    except Exception as e:
        logging.error("Tibber Quarter Request fehlgeschlagen: %s", e)
        return None

def update_price_cache(pi):
    today = dt.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi['today']}, CACHE_TODAY)

def cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def _filter_range_for_date(nodes, date_obj):
    out = []
    for n in nodes or []:
        try:
            ts = dt.datetime.fromisoformat(n["startsAt"]).astimezone(LOCAL_TZ)
        except Exception:
            continue
        if ts.date() == date_obj:
            out.append({"startsAt": n["startsAt"], "total": n["total"]})
    return out

def prepare_info(today_slots, current):
    today_vals = [s["total"] * 100 for s in today_slots]
    cur = current or {}
    cur_start = cur.get("startsAt") or dt.datetime.now(LOCAL_TZ).isoformat()
    cur_dt = dt.datetime.fromisoformat(cur_start).astimezone(LOCAL_TZ)
    cur_price = float(cur.get("total") or 0) * 100
    low_idx = int(np.argmin(today_vals)) if today_vals else None
    low_time = (
        dt.datetime.fromisoformat(today_slots[low_idx]["startsAt"]).astimezone(LOCAL_TZ)
        if low_idx is not None
        else None
    )
    return {
        "current_dt": cur_dt,
        "current_price": cur_price,
        "lowest_today": min(today_vals) if today_vals else 0,
        "lowest_today_time": low_time,
        "highest_today": max(today_vals) if today_vals else 0,
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


def slots_to_15min(slots):
    """
    Nutzt echte 15-Minuten-Slots, sofern die Tibber-Antwort diese liefert.
    Andernfalls wird wie bisher jede Stunde auf vier 15-Minuten-Segmente
    erweitert.
    """
    parsed = []
    for s in sorted(slots, key=lambda x: x.get("startsAt", "")):
        try:
            start = dt.datetime.fromisoformat(s["startsAt"]).astimezone(LOCAL_TZ)
            price = s["total"] * 100
            parsed.append((start, price))
        except Exception:
            continue

    if len(parsed) >= 2:
        deltas_min = [
            (parsed[i + 1][0] - parsed[i][0]).total_seconds() / 60.0
            for i in range(len(parsed) - 1)
        ]
        if min(deltas_min) <= 16:  # bereits 15-Minuten-Auflösung
            ts_list = [p[0] for p in parsed]
            val_list = [p[1] for p in parsed]
            return ts_list, val_list

    return expand_to_15min(slots)

def normalize_price_slots_15min(slots):
    if not slots:
        return []
    ts_list, val_list = slots_to_15min(slots)
    return [
        {"startsAt": ts.isoformat(), "total": val / 100.0}
        for ts, val in zip(ts_list, val_list)
    ]

def pick_current_price(quarter_range, fallback):
    for src in (quarter_range, fallback):
        cur = (src or {}).get("current") or {}
        if cur.get("startsAt") and cur.get("total") is not None:
            return cur
    return {"startsAt": dt.datetime.now(LOCAL_TZ).isoformat(), "total": 0.0}

# ---------- DB-Serien ----------
def series_from_db(table, column, slots_dt, max_age_hours=48):
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(f"SELECT ts, {column} FROM {table}", conn)
    except Exception:
        conn.close(); return None
    conn.close()

    if df.empty:
        return None

    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)

    if max_age_hours is not None:
        newest = df.index.max()
        if newest is None:
            return None
        if newest < dt.datetime.now(tz=LOCAL_TZ) - dt.timedelta(hours=max_age_hours):
            return None

    df = df.resample('15T').mean().ffill().fillna(0)
    out = []
    for t in slots_dt:
        v = df[column].asof(t) if not df.empty else 0.0
        out.append(float(0.0 if pd.isna(v) else v))
    return pd.Series(out, index=slots_dt)

def pv_series_from_db(slots_dt, column, db_file=DB_FILE):
    if not slots_dt:
        return pd.Series(dtype=float, index=slots_dt)
    if column not in ("pv1_w", "pv2_w", "pv_sum_w"):
        raise ValueError(f"PV DB column ungültig: {column}")

    start_local = min(slots_dt)
    end_local = max(slots_dt)
    start_utc = int((start_local - dt.timedelta(hours=1)).astimezone(dt.timezone.utc).timestamp())
    end_utc = int((end_local + dt.timedelta(hours=1)).astimezone(dt.timezone.utc).timestamp())

    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(
            f"SELECT ts, {column} FROM pv_log WHERE ts BETWEEN ? AND ?",
            conn,
            params=(start_utc, end_utc),
        )
    except Exception as e:
        conn.close()
        logging.warning("PV DB query failed (%s): %s", column, e)
        return pd.Series([np.nan] * len(slots_dt), index=slots_dt)
    conn.close()

    if df.empty:
        return pd.Series([np.nan] * len(slots_dt), index=slots_dt)

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("15T").mean()

    out = []
    for t in slots_dt:
        v = df[column].asof(t) if not df.empty else np.nan
        out.append(float(v) if v is not None and not pd.isna(v) else np.nan)
    return pd.Series(out, index=slots_dt)


def get_pv_series_db(slots_dt):
    series_map = {
        "pv1": pv_series_from_db(slots_dt, "pv1_w"),
        "pv2": pv_series_from_db(slots_dt, "pv2_w"),
        "pv_sum": pv_series_from_db(slots_dt, "pv_sum_w"),
    }
    for key, series in series_map.items():
        if series is None or series.empty or not np.isfinite(series.to_numpy()).any():
            series_map[key] = pd.Series([np.nan] * len(slots_dt), index=slots_dt)
        else:
            series_map[key] = _mask_future(series)
    return series_map


def pv_daily_energy_wh_from_db(date_local, db_file=DB_FILE):
    start_local = dt.datetime.combine(date_local, dt.time.min, tzinfo=LOCAL_TZ)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    start_utc = int(start_local.astimezone(dt.timezone.utc).timestamp())
    end_utc = int(end_local.astimezone(dt.timezone.utc).timestamp())

    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(
            "SELECT ts, pv_sum_w FROM pv_log WHERE ts BETWEEN ? AND ?",
            conn,
            params=(start_utc, end_utc),
        )
    except Exception as e:
        conn.close()
        logging.warning("PV daily energy query failed (%s): %s", date_local, e)
        return None
    conn.close()

    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("15T").mean().ffill().fillna(0)
    energy_wh = float((df["pv_sum_w"] * 0.25).sum())
    return energy_wh


def pv_profile_normalized_from_db(date_local, slots_dt_template, db_file=DB_FILE):
    start_local = dt.datetime.combine(date_local, dt.time.min, tzinfo=LOCAL_TZ)
    end_local = start_local + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    start_utc = int(start_local.astimezone(dt.timezone.utc).timestamp())
    end_utc = int(end_local.astimezone(dt.timezone.utc).timestamp())

    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(
            "SELECT ts, pv_sum_w FROM pv_log WHERE ts BETWEEN ? AND ?",
            conn,
            params=(start_utc, end_utc),
        )
    except Exception as e:
        conn.close()
        logging.warning("PV profile query failed (%s): %s", date_local, e)
        return None
    conn.close()

    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("15T").mean().ffill().fillna(0)
    profile = df["pv_sum_w"].reindex(slots_dt_template, method="ffill").fillna(0)
    profile = profile.clip(lower=0)
    energy_wh = float((profile * 0.25).sum())
    if energy_wh < 50:
        return None
    total_w = float(profile.sum())
    if total_w <= 0:
        return None
    profile_norm = profile / total_w
    return pd.Series(profile_norm, index=slots_dt_template)


def pv_forecast_series_for_date(target_date, slots_dt):
    if not slots_dt:
        return pd.Series(dtype=float, index=slots_dt)
    slots_dt_template = slots_dt
    profiles = []
    for days_back in range(1, 15):
        day = target_date - dt.timedelta(days=days_back)
        shifted_slots = [
            dt.datetime.combine(day, slot.astimezone(LOCAL_TZ).timetz())
            for slot in slots_dt_template
        ]
        p = pv_profile_normalized_from_db(day, shifted_slots)
        if p is not None and len(p) == len(slots_dt_template):
            p = p.copy()
            p.index = slots_dt_template
            profiles.append(p)

    if not profiles:
        return pd.Series([np.nan] * len(slots_dt), index=slots_dt)

    profile_df = pd.concat(profiles, axis=1)
    median_profile = profile_df.median(axis=1)

    energies = []
    for days_back in range(1, 8):
        day = target_date - dt.timedelta(days=days_back)
        e_wh = pv_daily_energy_wh_from_db(day)
        if e_wh is not None and e_wh > 50:
            energies.append(e_wh)
    base_wh = float(np.mean(energies)) if energies else 0.0

    sun_today = globals().get("SUN_TODAY")
    sun_tomorrow = globals().get("SUN_TOMORROW")
    scale = 1.0
    if sun_today is not None and sun_tomorrow is not None:
        denom = max(float(sun_today), 0.2)
        scale = float(sun_tomorrow) / denom
        scale = max(0.2, min(1.8, scale))

    forecast_wh = base_wh * scale
    median_sum = float(median_profile.sum())
    if median_sum <= 0:
        return pd.Series([np.nan] * len(slots_dt), index=slots_dt)
    median_profile = median_profile / max(median_sum, 1e-9)
    w_series = median_profile * (forecast_wh / 0.25)
    return pd.Series(w_series, index=slots_dt)

def pv_db_stats(slots_dt, label, db_file=DB_FILE):
    if not slots_dt:
        logging.info("PV DB empty for range; skipping PV lines")
        return {"count": 0, "max_pv1": None, "max_pv2": None, "max_pv_sum": None}

    start_local = min(slots_dt)
    end_local = max(slots_dt)
    start_utc = int((start_local - dt.timedelta(hours=1)).astimezone(dt.timezone.utc).timestamp())
    end_utc = int((end_local + dt.timedelta(hours=1)).astimezone(dt.timezone.utc).timestamp())

    conn = sqlite3.connect(db_file)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt,
                   MAX(pv1_w) as max_pv1,
                   MAX(pv2_w) as max_pv2,
                   MAX(pv_sum_w) as max_pv_sum
            FROM pv_log
            WHERE ts BETWEEN ? AND ?
            """,
            (start_utc, end_utc),
        ).fetchone()
    except Exception as e:
        conn.close()
        logging.warning("PV DB stats query failed (%s): %s", label, e)
        return {"count": 0, "max_pv1": None, "max_pv2": None, "max_pv_sum": None}
    conn.close()

    count = int(row[0] or 0)
    stats = {
        "count": count,
        "max_pv1": row[1],
        "max_pv2": row[2],
        "max_pv_sum": row[3],
    }
    if count == 0:
        logging.info("PV DB empty for range; skipping PV lines")
    else:
        logging.info(
            "PV DB rows %s: %d (max pv1=%s pv2=%s pv_sum=%s)",
            label,
            count,
            stats["max_pv1"],
            stats["max_pv2"],
            stats["max_pv_sum"],
        )
    return stats


def _mask_future(series):
    """Setzt zukünftige Slots auf NaN, damit keine flache Linie bis Mitternacht gezeichnet wird."""
    if series is None or series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return series
    now = dt.datetime.now(tz=LOCAL_TZ)
    masked = series.copy()
    masked[masked.index > now] = np.nan
    return masked


def get_consumption_series(slots_dt):
    return series_from_db("consumption_log", "consumption_w", slots_dt)

_PV_LOGGED = False

def get_pv_series_micro(slots_dt, micro_sn, code, kind="auto"):
    if not slots_dt or not micro_sn or not code:
        return _mask_future(pd.Series([0.0] * len(slots_dt), index=slots_dt))
    target_date = slots_dt[0].date()
    begin_local = dt.datetime.combine(target_date, dt.time.min, tzinfo=LOCAL_TZ)
    end_local = dt.datetime.combine(target_date, dt.time.max, tzinfo=LOCAL_TZ)
    hist = ecoflow_quota_data(micro_sn, begin_local, end_local, code, expected_kind=kind)
    values = []
    now = dt.datetime.now(tz=LOCAL_TZ)
    for ts in slots_dt:
        if ts > now:
            values.append(np.nan)
            continue
        v = hist.asof(ts) if hist is not None and not hist.empty else None
        values.append(float(0.0 if v is None or pd.isna(v) else v))
    series = pd.Series(values, index=slots_dt)
    series.attrs["energy_to_power"] = bool(hist.attrs.get("energy_to_power")) if hist is not None else False
    return _mask_future(series)

def get_pv_total_series_micro(slots_dt):
    if not slots_dt:
        return _mask_future(pd.Series(dtype=float))
    micro_sn = getattr(api_key, "ECOFLOW_MIKRO_ID", "").strip()
    if not micro_sn:
        return _mask_future(pd.Series([0.0] * len(slots_dt), index=slots_dt))
    try:
        today = dt.datetime.now(tz=LOCAL_TZ).date()
        begin_local = dt.datetime.combine(today, dt.time.min, tzinfo=LOCAL_TZ)
        end_local = dt.datetime.now(tz=LOCAL_TZ)
        hist = ecoflow_quota_data(micro_sn, begin_local, end_local, ECOFLOW_SOLAR_ENERGY_CODE, expected_kind="auto")
        if hist is None or hist.empty:
            series = pd.Series([np.nan] * len(slots_dt), index=slots_dt)
            series.attrs["energy_to_power"] = False
            series.attrs["pv_missing"] = True
            logging.info("PV energy->power triggered: %s", series.attrs.get("energy_to_power"))
            return _mask_future(series)
        values = []
        now = dt.datetime.now(tz=LOCAL_TZ)
        for ts in slots_dt:
            if ts > now:
                values.append(np.nan)
                continue
            v = hist.asof(ts) if hist is not None and not hist.empty else None
            values.append(float(0.0 if v is None or pd.isna(v) else v))
        series = pd.Series(values, index=slots_dt)
        series.attrs["energy_to_power"] = bool(hist.attrs.get("energy_to_power")) if hist is not None else False
        logging.info("PV energy->power triggered: %s", series.attrs.get("energy_to_power"))
        return _mask_future(series)
    except Exception as e:
        logging.error("EcoFlow PV total series fehlgeschlagen: %s", e)
        return _mask_future(pd.Series([0.0] * len(slots_dt), index=slots_dt))

def get_pv_series_multi_micro(slots_dt):
    global _PV_LOGGED
    micro_sn = getattr(api_key, "ECOFLOW_MIKRO_ID", "").strip()
    pv1 = None
    pv2 = None
    pv_sum = get_pv_total_series_micro(slots_dt)

    pv_sum = _mask_future(pv_sum)

    if not _PV_LOGGED:
        _PV_LOGGED = True
        logging.info(
            "PV Codes (Micro): PV1=%s PV2=%s PV_TOTAL=%s",
            ECOFLOW_PV1_CODE,
            ECOFLOW_PV2_CODE,
            ECOFLOW_PV_TOTAL_CODE or "-",
        )
        logging.info(
            "PV Serienpunkte: pv1=%d pv2=%d pv_sum=%d",
            len(pv1) if pv1 is not None else 0,
            len(pv2) if pv2 is not None else 0,
            len(pv_sum) if pv_sum is not None else 0,
        )
        logging.info(
            "PV energy->power triggered: pv1=%s pv2=%s pv_sum=%s",
            bool(pv1.attrs.get("energy_to_power")) if pv1 is not None else False,
            bool(pv2.attrs.get("energy_to_power")) if pv2 is not None else False,
            bool(pv_sum.attrs.get("energy_to_power")) if pv_sum is not None else False,
        )

    return pv1, pv2, pv_sum

# ---------- Tibber Consumption (hourly -> 15min) ----------
def tibber_hourly_consumption(last=48):
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
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
    homes = (((j.get("data") or {}).get("viewer") or {}).get("homes") or [])
    home = pick_home_with_data(homes) or {}
    cons = (home.get("consumption") or {})
    nodes = (cons.get("nodes") or []) if isinstance(cons, dict) else []
    if not nodes:
        logging.info("Tibber Consumption leer/fehlend: homes=%d", len(homes))
        return []
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
def fetch_openmeteo_hourly(lat, lon):
    """
    Holt Open-Meteo stündliche Wettercodes + Tag/Nacht.
    Return: {datetime_local_hour: (code:int, is_day:bool)}
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=weathercode,is_day"
            "&timezone=Europe%2FBerlin"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        hourly = j.get("hourly", {}) or {}
        times = hourly.get("time") or []
        codes = hourly.get("weathercode") or []
        is_day_list = hourly.get("is_day") or []
        hourly_map = {}
        for t_str, code, is_day in zip(times, codes, is_day_list):
            try:
                t = dt.datetime.fromisoformat(t_str)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=LOCAL_TZ)
                hourly_map[t] = (int(code), bool(is_day))
            except Exception:
                continue
        return hourly_map
    except Exception as e:
        logging.error("Open-Meteo hourly fetch failed: %s", e)
        return {}

def fetch_openmeteo_sunshine_hours(lat, lon):
    """
    Holt Open-Meteo daily sunshine_duration (Sekunden) für heute und morgen.
    Return: (sun_today_h, sun_tomorrow_h) in Stunden oder (None, None)
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=sunshine_duration"
            "&timezone=Europe%2FBerlin"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily", {}) or {}
        arr = daily.get("sunshine_duration") or []
        if len(arr) < 2:
            raise ValueError("sunshine_duration missing or incomplete")
        sun_today_h = float(arr[0]) / 3600.0
        sun_tomorrow_h = float(arr[1]) / 3600.0
        return sun_today_h, sun_tomorrow_h
    except Exception as e:
        logging.error("Open-Meteo sunshine fetch failed: %s", e)
        return None, None

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

def ecoflow_get_main_sn(sn_any):
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/system/main/sn"
    query = {"sn": sn_any}
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, query, content_type=None)
    url = f"{base}{path}?{urlencode(query)}"
    r = requests.get(url, headers=hdr, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    main_sn = None
    if r.status_code == 200 and str(j.get("code")) == "0":
        data = j.get("data") if isinstance(j, dict) else None
        if isinstance(data, dict):
            main_sn = data.get("sn") or data.get("mainSn") or data.get("deviceSn")
        elif isinstance(data, str):
            main_sn = data
    logging.info(
        "EcoFlow main-sn resolve: input=%s -> main=%s (http=%s, code=%s)",
        sn_any,
        main_sn,
        r.status_code,
        j.get("code") if isinstance(j, dict) else None,
    )
    return main_sn

def ecoflow_get_all_quota(sn, with_status=False):
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

    data = j.get("data") if isinstance(j, dict) else None
    data_type = type(data).__name__
    try:
        data_len = len(data) if isinstance(data, (dict, list)) else 0
    except Exception:
        data_len = 0
    logging.info(
        "EcoFlow RAW quota/all sn=%s http=%s code=%s msg=%s data_type=%s data_len=%s",
        sn,
        r.status_code,
        j.get("code") if isinstance(j, dict) else None,
        j.get("message") if isinstance(j, dict) else None,
        data_type,
        data_len,
    )

    status_label = "quota/all"
    if r.status_code == 200 and str(j.get("code")) == "0":
        data = j.get("data", {}) or {}
        return (data, status_label) if with_status else data

    # Einige BKW/Stream-Geräte liefern die Daten nur über /device/quota,
    # deshalb probieren wir dieses Fallback automatisch.
    alt_path = "/iot-open/sign/device/quota"
    alt_url = f"{base}{alt_path}?{urlencode(query)}"
    r_alt = requests.get(alt_url, headers=hdr, timeout=12)
    try:
        j_alt = r_alt.json()
    except Exception:
        j_alt = {"raw": r_alt.text}

    data_alt = j_alt.get("data") if isinstance(j_alt, dict) else None
    data_alt_type = type(data_alt).__name__
    try:
        data_alt_len = len(data_alt) if isinstance(data_alt, (dict, list)) else 0
    except Exception:
        data_alt_len = 0
    logging.info(
        "EcoFlow RAW quota/fallback sn=%s http=%s code=%s msg=%s data_type=%s data_len=%s",
        sn,
        r_alt.status_code,
        j_alt.get("code") if isinstance(j_alt, dict) else None,
        j_alt.get("message") if isinstance(j_alt, dict) else None,
        data_alt_type,
        data_alt_len,
    )
    if r_alt.status_code == 200 and str(j_alt.get("code")) == "0":
        status_label = "quota fallback"
        data = j_alt.get("data", {}) or {}
        return (data, status_label) if with_status else data

    raise RuntimeError(
        "EcoFlow quota fehlgeschlagen: primary %s resp=%s | fallback %s resp=%s" % (
            f"HTTP {r.status_code}", str(j)[:200], f"HTTP {r_alt.status_code}", str(j_alt)[:200]
        )
    )


def ecoflow_get_quota_selected(sn, quotas: list[str]) -> dict:
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/quota"
    query = {"sn": sn}
    body = {
        "sn": sn,
        "params": {
            "quotas": quotas,
        },
    }
    hdr = _signed_headers(
        api_key.ECOFLOW_APP_KEY,
        api_key.ECOFLOW_SECRET_KEY,
        body,
        content_type="application/json",
    )
    url = f"{base}{path}?{urlencode(query)}"
    r = requests.post(url, headers=hdr, json=body, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    data = j.get("data") if isinstance(j, dict) else None
    data_type = type(data).__name__
    try:
        data_len = len(data) if isinstance(data, (dict, list)) else 0
    except Exception:
        data_len = 0
    logging.info(
        "EcoFlow selected quota sn=%s http=%s code=%s msg=%s data_type=%s data_len=%s",
        sn,
        r.status_code,
        j.get("code") if isinstance(j, dict) else None,
        j.get("message") if isinstance(j, dict) else None,
        data_type,
        data_len,
    )

    if r.status_code == 200 and str(j.get("code")) == "0":
        try:
            if data is None:
                logging.info("EcoFlow selected quota raw data=None")
            elif isinstance(data, dict):
                compact_parts = []
                for dk, dv in data.items():
                    try:
                        if isinstance(dv, dict):
                            compact_parts.append(f"{dk}=<dict len={len(dv)}>")
                        elif isinstance(dv, list):
                            compact_parts.append(f"{dk}=<list len={len(dv)}>")
                        else:
                            compact_parts.append(f"{dk}={dv}")
                    except Exception:
                        continue
                logging.info("EcoFlow selected quota raw data: %s", "; ".join(compact_parts))
            else:
                logging.info("EcoFlow selected quota raw data: %s", data)
        except Exception:
            try:
                logging.info("EcoFlow selected quota raw data logging failed")
            except Exception:
                pass
        return data or {}
    raise RuntimeError(f"EcoFlow selected quota fehlgeschlagen: HTTP {r.status_code}, resp={str(j)[:200]}")


def ecoflow_get_quota_selected_get(sn, quotas: list[str]) -> dict:
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/quota"
    params_dict = {
        "sn": sn,
        "params": {
            "quotas": quotas,
        },
    }
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, params_dict, content_type=None)
    query_pairs = sorted(_flatten_params(params_dict), key=lambda kv: kv[0])
    url = f"{base}{path}?{urlencode(query_pairs)}"
    r = requests.get(url, headers=hdr, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    data = j.get("data") if isinstance(j, dict) else None
    data_type = type(data).__name__
    try:
        data_len = len(data) if isinstance(data, (dict, list)) else 0
    except Exception:
        data_len = 0
    logging.info(
        "EcoFlow RAW quota(GET) sn=%s http=%s code=%s msg=%s data_type=%s data_len=%s",
        sn,
        r.status_code,
        j.get("code") if isinstance(j, dict) else None,
        j.get("message") if isinstance(j, dict) else None,
        data_type,
        data_len,
    )

    if r.status_code == 200 and str(j.get("code")) == "0":
        try:
            if data is None:
                logging.info("EcoFlow selected quota raw data=None")
            elif isinstance(data, dict):
                compact_parts = []
                for dk, dv in data.items():
                    try:
                        if isinstance(dv, dict):
                            compact_parts.append(f"{dk}=<dict len={len(dv)}>")
                        elif isinstance(dv, list):
                            compact_parts.append(f"{dk}=<list len={len(dv)}>")
                        else:
                            compact_parts.append(f"{dk}={dv}")
                    except Exception:
                        continue
                logging.info("EcoFlow selected quota raw data: %s", "; ".join(compact_parts))
            else:
                logging.info("EcoFlow selected quota raw data: %s", data)
        except Exception:
            try:
                logging.info("EcoFlow selected quota raw data logging failed")
            except Exception:
                pass
        return data or {}
    raise RuntimeError(f"EcoFlow selected quota(GET) fehlgeschlagen: HTTP {r.status_code}, resp={str(j)[:200]}")


def _parse_ecoflow_quota_data_payload(j):
    if not isinstance(j, dict):
        return None
    data = j.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data.get("data")
        if isinstance(data.get("list"), list):
            return data.get("list")
        if isinstance(data.get("records"), list):
            return data.get("records")
        if isinstance(data.get("quotaData"), list):
            return data.get("quotaData")
    return None


def _coerce_timestamp(ts_raw):
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)) or str(ts_raw).isdigit():
            ts_int = int(float(ts_raw))
            if ts_int > 1e12:  # ms
                ts_dt = dt.datetime.fromtimestamp(ts_int / 1000.0, tz=dt.timezone.utc)
            else:
                ts_dt = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc)
            return ts_dt.astimezone(LOCAL_TZ)
        ts_str = str(ts_raw).replace("T", " ").strip()
        ts_dt = dt.datetime.fromisoformat(ts_str)
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=dt.timezone.utc)
        return ts_dt.astimezone(LOCAL_TZ)
    except Exception:
        return None


def ecoflow_quota_data(sn_any, begin_dt_local, end_dt_local, code, expected_kind="auto"):
    if not code:
        logging.warning("EcoFlow quota/data: code fehlt -> leere Serie")
        return pd.Series(dtype=float)
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path_main = "/iot-open/sign/device/system/main/sn"
    try:
        hdr_main = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, {"sn": sn_any})
        url_main = f"{base}{path_main}?{urlencode({'sn': sn_any})}"
        r_main = requests.get(url_main, headers=hdr_main, timeout=12)
        j_main = r_main.json()
        main_sn = safe_get(j_main, "data", default=None)
        if isinstance(main_sn, dict):
            main_sn = main_sn.get("sn") or main_sn.get("mainSn") or main_sn.get("deviceSn")
        elif not isinstance(main_sn, str):
            main_sn = None
    except Exception:
        main_sn = None
    if not main_sn:
        main_sn = sn_any

    def _ensure_local(dt_like):
        if dt_like.tzinfo is None:
            return dt_like.replace(tzinfo=LOCAL_TZ)
        return dt_like.astimezone(LOCAL_TZ)

    begin_local = _ensure_local(begin_dt_local)
    end_local = _ensure_local(end_dt_local)
    begin_utc = begin_local.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_utc = end_local.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _fetch_quota_data(begin_val, end_val, mode_label):
        body = {
            "sn": main_sn,
            "params": {
                "beginTime": begin_val,
                "endTime": end_val,
                "code": code,
            },
        }
        hdr = _signed_headers(
            api_key.ECOFLOW_APP_KEY,
            api_key.ECOFLOW_SECRET_KEY,
            body,
            content_type="application/json;charset=UTF-8",
        )
        url = f"{base}/iot-open/sign/device/quota/data"
        try:
            r = requests.post(url, headers=hdr, json=body, timeout=15)
            try:
                j = r.json()
            except Exception:
                j = {"raw": r.text}
            entries = _parse_ecoflow_quota_data_payload(j) or []
            logging.info(
                "EcoFlow quota/data %s http=%s code=%s msg=%s entries=%s",
                mode_label,
                r.status_code,
                j.get("code") if isinstance(j, dict) else None,
                j.get("message") if isinstance(j, dict) else None,
                len(entries),
            )
            return entries, j
        except Exception as e:
            logging.error("EcoFlow quota/data request failed (%s): %s", mode_label, e)
            return [], {"error": str(e)}

    entries, j = _fetch_quota_data(begin_utc, end_utc, "utc")
    if not entries:
        _dump_json_force(f"ecoflow_quota_data_{main_sn}_{code}_utc", j)
        begin_local_str = begin_local.strftime("%Y-%m-%d %H:%M:%S")
        end_local_str = end_local.strftime("%Y-%m-%d %H:%M:%S")
        entries, j = _fetch_quota_data(begin_local_str, end_local_str, "local")
        _dump_json_force(f"ecoflow_quota_data_{main_sn}_{code}_local", j)
    if not entries:
        begin_epoch = int(begin_local.timestamp() * 1000)
        end_epoch = int(end_local.timestamp() * 1000)
        entries, j = _fetch_quota_data(begin_epoch, end_epoch, "epoch")
        _dump_json_force(f"ecoflow_quota_data_{main_sn}_{code}_epoch", j)

    points = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ts_dt = _coerce_timestamp(
            entry.get("timestamp")
            or entry.get("time")
            or entry.get("ts")
            or entry.get("timeStamp")
            or entry.get("dateTime")
            or entry.get("datetime")
        )
        if ts_dt is None:
            continue
        val = (
            _to_float(entry.get("value"))
            or _to_float(entry.get("val"))
            or _to_float(entry.get("data"))
            or _to_float(entry.get("energy"))
            or _to_float(entry.get("power"))
            or _to_float(entry.get("indexValue"))
        )
        if val is None:
            continue
        points.append((ts_dt, float(val)))

    if not points:
        logging.warning("WARNING: quota/data returned empty")
        return pd.Series(dtype=float)

    df = pd.DataFrame(points, columns=["ts", "value"])
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("15T").mean().ffill()
    series = df["value"]
    energy_to_power = False

    def _to_power(s):
        return s.diff().fillna(0).clip(lower=0) * 4.0

    if expected_kind == "energy":
        series = _to_power(series)
        energy_to_power = True
    elif expected_kind == "power":
        energy_to_power = False
    else:
        try:
            diffs = series.diff().dropna()
            if len(diffs) >= 3:
                non_neg_frac = float((diffs >= 0).mean())
                if series.max() > 100 and non_neg_frac >= 0.7:
                    series = _to_power(series)
                    energy_to_power = True
        except Exception:
            pass

    series.attrs["energy_to_power"] = energy_to_power

    logging.info(
        "EcoFlow quota/data main_sn=%s code=%s points=%s first=%s last=%s energy_to_power=%s",
        main_sn,
        code,
        len(series),
        series.index.min(),
        series.index.max(),
        energy_to_power,
    )
    return series

def ecoflow_status_bkw():
    """
    Mapping für EcoFlow Stream/PowerStream basierend auf deinen Keys.
    - soc: cmsBattSoc (%)
    - pv_input_w_sum: powGetPvSum (W)
    - load_w: powGetSysLoad (W)
    - grid_w: powGetSysGrid (W) oder gridConnectionPower (W)
    - power_w: bevorzugt aus powGetBpCms (umgedrehtes Vorzeichen), sonst Bilanz = load - pv - grid
               (>0 Entladen, <0 Laden)
    - mode: kompakt aus feedGridMode + energyStrategyOperateMode.*
    - eta_min: bei Stream meist nicht vorhanden -> None
    Zusätzlich: Rohdaten-Dump nach ecoflow_quota_last.json.
    """
    sn_main = getattr(api_key, "ECOFLOW_DEVICE_ID", "").strip()
    sn_micro = getattr(api_key, "ECOFLOW_MIKRO_ID", "").strip()
    if not sn_main and not sn_micro:
        raise RuntimeError("ECOFLOW_DEVICE_ID oder ECOFLOW_MIKRO_ID fehlt in api_key.py")

    sn_for_resolve = sn_main or sn_micro
    sn_main_effective = sn_for_resolve
    try:
        resolved = ecoflow_get_main_sn(sn_for_resolve)
        if resolved:
            sn_main_effective = resolved
    except Exception as e:
        logging.info("EcoFlow main-sn lookup failed: %s", e)

    logging.info(
        "EcoFlow SNs: configured_main=%s, micro=%s, resolved_main=%s",
        sn_main or "-",
        sn_micro or "-",
        sn_main_effective or "-",
    )

    device_map = {}
    try:
        devices = ecoflow_get_device_list()
        if isinstance(devices, list):
            for dev in devices:
                try:
                    if not isinstance(dev, dict):
                        continue
                    sn = dev.get("sn") or dev.get("deviceSn")
                    if not sn:
                        continue
                    name = dev.get("deviceName") or dev.get("name") or "-"
                    model = dev.get("model") or dev.get("productName") or "-"
                    dtype = dev.get("deviceType") or dev.get("type") or "-"
                    online = dev.get("online")
                    device_map[sn] = {
                        "name": name or "-",
                        "model": model or "-",
                        "type": dtype or "-",
                        "online": online,
                    }
                except Exception:
                    continue
        logging.info("=== EcoFlow device/list ===")
        try:
            logging.info("- Anzahl devices: %s", len(devices))
        except Exception:
            logging.info("- Anzahl devices: ?")
        info_main = device_map.get(sn_main_effective)
        logging.info(
            "Batterie SN=%s in_list=%s name=%s type=%s",
            sn_main_effective or "-",
            "yes" if sn_main_effective and sn_main_effective in device_map else "no",
            (info_main or {}).get("name", "-"),
            (info_main or {}).get("type", "-"),
        )
        info_micro = device_map.get(sn_micro)
        logging.info(
            "Wechselrichter SN=%s in_list=%s name=%s type=%s",
            sn_micro or "-",
            "yes" if sn_micro and sn_micro in device_map else "no",
            (info_micro or {}).get("name", "-"),
            (info_micro or {}).get("type", "-"),
        )
    except Exception as e:
        logging.info("device/list ERROR: %s", e)

    q_main, q_pv = {}, {}
    main_status = None
    micro_status = None
    try:
        if sn_main_effective:
            if ECO_DEBUG:
                q_main, main_status = ecoflow_get_all_quota(sn_main_effective, with_status=True)
            else:
                q_main = ecoflow_get_all_quota(sn_main_effective)
                main_status = "quota/all"
            if ECO_DEBUG:
                logging.info(
                    "EcoFlow quota (%s) für Batterie/System %s",
                    main_status or "?",
                    sn_main_effective,
                )
            try:
                if sn_micro and sn_micro != sn_main_effective:
                    if ECO_DEBUG:
                        q_pv, micro_status = ecoflow_get_all_quota(sn_micro, with_status=True)
                    else:
                        q_pv = ecoflow_get_all_quota(sn_micro)
                        micro_status = "quota/all"
                    logging.info(
                        "EcoFlow quota (%s) für Mikro %s",
                        micro_status or "?",
                        sn_micro,
                    )
            except Exception as e:
                logging.info("EcoFlow quota/all für Mikro fehlgeschlagen: %s", e)
            try:
                if ECO_DEBUG:
                    with open("/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_quota_last.json", "w") as f:
                        json.dump({"main": q_main, "micro": q_pv}, f, indent=2)
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
        logging.info("EcoFlow Fallback-Datei wurde nicht genutzt oder existiert nicht")
        return {
            "soc": None, "power_w": None, "mode": None, "eta_min": None,
            "pv_input_w_sum": None, "pv1_input_w": None, "pv2_input_w": None,
            "grid_w": None, "load_w": None
        }

    try:
        logging.info("=== Batterie/System (HTTP Main-SN) ===")
        logging.info("- SN: %s", sn_main_effective or "-")
        try:
            logging.info("- Keys: %s", len(q_main) if isinstance(q_main, dict) else 0)
        except Exception:
            logging.info("- Keys: ?")
        batt_keys = [
            "cmsBattSoc", "powGetBpCms", "powGetPvSum", "powGetPv1InputW", "powGetPv2InputW",
            "powGetSysLoad", "powGetSysGrid", "gridConnectionPower", "feedGridMode",
            "energyStrategyOperateMode.operateSelfPoweredOpen",
            "energyStrategyOperateMode.operateIntelligentScheduleModeOpen",
            "chargePower", "dischargePower", "remainingChargeTimeMins", "remainingDischargeTimeMins",
        ]
        for bk in batt_keys:
            if isinstance(q_main, dict) and bk in q_main:
                logging.info("  %s = %s", bk, q_main.get(bk))

        logging.info("=== Mikro (HTTP) ===")
        if sn_micro:
            info_micro = device_map.get(sn_micro, {})
            logging.info(
                "- SN=%s name=%s online=%s",
                sn_micro,
                info_micro.get("name", "-"),
                info_micro.get("online"),
            )
        else:
            logging.info("- Kein Mikro-SN konfiguriert")
        if not q_pv:
            logging.info("Micro per-device quotas via HTTP not available; use MQTT quota topic if needed")
    except Exception:
        pass

    def num(src, key, default=None):
        v = src.get(key) if src else None
        fv = _to_float(v)
        return fv if fv is not None else default

    def pv_from_micro(q):
        if not isinstance(q, dict):
            return None
        v1, c1 = num(q, "pv1InputVolt"), num(q, "pv1InputCur")
        v2, c2 = num(q, "pv2InputVolt"), num(q, "pv2InputCur")

        def _p(v, c):
            if v is None or c is None:
                return 0.0
            p = (float(v) * float(c)) / 100.0
            if p < 0:
                return 0.0
            return p

        total = _p(v1, c1) + _p(v2, c2)
        if total <= 0:
            return None
        if total > 3000:
            logging.info("PV Micro-Ausreißer erkannt (%.1f W) -> clamp", total)
            total = 3000.0
        return total

    # Kerngrößen
    soc     = num(q_main, "cmsBattSoc")
    if soc is not None:
        try: soc = int(round(soc))
        except: pass

    pv_sum  = pv_from_micro(q_pv)
    if pv_sum is not None:
        logging.info("PV-Leistung aus Micro-Quotas genutzt: %.1f W", pv_sum)
    else:
        pv_sum = num(q_main, "powGetPvSum")
        if pv_sum is None:
            pv_sum = num(q_main, "powGetPv1InputW", default=None)
            if pv_sum is not None:
                pv_sum += num(q_main, "powGetPv2InputW", default=0.0)
        logging.info("PV-Leistung aus Batterie/System-Quotas genutzt: %s", pv_sum if pv_sum is not None else "None")
    load_w  = num(q_main, "powGetSysLoad")
    grid_w  = num(q_main, "powGetSysGrid")
    if grid_w is None:
        grid_w = num(q_main, "gridConnectionPower")

    # Batterie-Leistung
    bp      = num(q_main, "powGetBpCms")     # beobachtet: negativ bei Entladen
    if bp is not None:
        power_w = -bp                # Konvention: >0 Entladen, <0 Laden
    else:
        power_w = None
        if (load_w is not None) and (pv_sum is not None) and (grid_w is not None):
            power_w = (load_w - pv_sum - grid_w)

    # Modus kompakt
    feed_mode = q_main.get("feedGridMode")  # 0/1
    es_self   = q_main.get("energyStrategyOperateMode.operateSelfPoweredOpen")
    es_sched  = q_main.get("energyStrategyOperateMode.operateIntelligentScheduleModeOpen")
    mode_parts = []
    if feed_mode is not None:
        try: mode_parts.append(f"Feed:{int(float(feed_mode))}")
        except: mode_parts.append(f"Feed:{feed_mode}")
    if es_self is not None:  mode_parts.append(f"Self:{es_self}")
    if es_sched is not None: mode_parts.append(f"Sched:{es_sched}")
    mode = " | ".join(map(str, mode_parts)) if mode_parts else None

    return {
        "soc": soc,
        "power_w": power_w,
        "mode": mode,
        "eta_min": None,
        "pv_input_w_sum": pv_sum,
        "pv1_input_w": None,
        "pv2_input_w": None,
        "grid_w": grid_w,
        "load_w": load_w,
        "micro_online": (device_map.get(sn_micro, {}) or {}).get("online")
    }



# ---------- Drawing ----------
def draw_dashed_line(d, x1, y1, x2, y2, dash=2, gap=2, fill=0, width=1):
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        return
    dx, dy = x2-x1, y2-y1
    dist = math.hypot(dx, dy)
    if dist == 0 or not math.isfinite(dist):
        return
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

def _text_size(d, text, font):
    bbox = d.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def _s(value, scale):
    return int(round(value * scale))

def _text(d, x, y, text, font, fill=0):
    d.text((int(round(x)), int(round(y))), text, font=font, fill=fill)


def _fmt_hours(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"


_BAYER_MATRIX_CACHE = {}

def _build_bayer_matrix(size):
    if size == 2:
        return [[0, 2], [3, 1]]
    if size % 2 != 0:
        raise ValueError("Bayer size must be power of two.")
    half = size // 2
    prev = _build_bayer_matrix(half)
    out = [[0 for _ in range(size)] for _ in range(size)]
    for y in range(half):
        for x in range(half):
            v = prev[y][x] * 4
            out[y][x] = v
            out[y][x + half] = v + 2
            out[y + half][x] = v + 3
            out[y + half][x + half] = v + 1
    return out

def _get_bayer_matrix(size):
    if size not in _BAYER_MATRIX_CACHE:
        _BAYER_MATRIX_CACHE[size] = _build_bayer_matrix(size)
    return _BAYER_MATRIX_CACHE[size]

def render_to_epd(img_l):
    """Finale Dither-Konvertierung nach 1-bit."""
    if img_l.mode != "L":
        img_l = img_l.convert("L")
    if GAMMA and CONTRAST and (GAMMA != 1.0 or CONTRAST != 1.0):
        lut = []
        gamma = float(GAMMA)
        contrast = float(CONTRAST)
        for i in range(256):
            v = i / 255.0
            if gamma != 1.0:
                v = pow(v, 1.0 / gamma)
            if contrast != 1.0:
                v = (v - 0.5) * contrast + 0.5
            v = max(0.0, min(1.0, v))
            lut.append(int(round(v * 255)))
        img_l = img_l.point(lut)
    if DITHER_MODE == "FS":
        return img_l.convert("1", dither=Image.FLOYDSTEINBERG)
    matrix = _get_bayer_matrix(16 if BAYER_MATRIX == 16 else 8)
    size = len(matrix)
    threshold_scale = 255.0 / (size * size)
    w, h = img_l.size
    out = Image.new("1", (w, h), 1)
    src = img_l.load()
    dst = out.load()
    for y in range(h):
        row = matrix[y % size]
        for x in range(w):
            threshold = (row[x % size] + 0.5) * threshold_scale
            dst[x, y] = 0 if src[x, y] < threshold else 255
    return out


def _smooth_series(series, window=3):
    if series is None:
        return None
    try:
        return series.rolling(window=window, center=True, min_periods=1).mean()
    except Exception:
        return series

def meteo_bucket(code):
    try:
        code = int(code)
    except Exception:
        return "cloudy"
    if code == 0:
        return "clear"
    if code in (1, 2):
        return "partly"
    if code == 3:
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "thunder"
    return "cloudy"


def draw_weather_icon(draw, x, y, size, code, is_day, fill=0):
    scale = max(1.0, float(size) / 40.0)
    width = max(1, int(round(1.0 * scale)))
    stroke = fill

    def sx(v):
        return int(round(x + v * scale))

    def sy(v):
        return int(round(y + v * scale))

    def draw_sun(cx, cy, r):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=stroke, width=width)
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            x1 = cx + math.cos(rad) * (r + 4 * scale)
            y1 = cy + math.sin(rad) * (r + 4 * scale)
            x2 = cx + math.cos(rad) * (r + 10 * scale)
            y2 = cy + math.sin(rad) * (r + 10 * scale)
            draw.line((x1, y1, x2, y2), fill=stroke, width=width)

    def draw_moon(cx, cy, r):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=stroke, width=width)
        cut_r = int(r * 0.9)
        draw.ellipse((cx - cut_r + int(3 * scale), cy - cut_r,
                      cx + cut_r + int(3 * scale), cy + cut_r), fill=255, outline=255)

    def draw_cloud(cx, cy, w, h):
        r1 = int(w * 0.22)
        r2 = int(w * 0.26)
        r3 = int(w * 0.20)
        base_y = cy + int(h * 0.55)
        draw.ellipse((cx - int(w * 0.4), base_y - r1, cx - int(w * 0.4) + 2 * r1,
                      base_y + r1), outline=stroke, width=width)
        draw.ellipse((cx - int(w * 0.1), base_y - r2 - int(h * 0.2), cx - int(w * 0.1) + 2 * r2,
                      base_y + r2 - int(h * 0.2)), outline=stroke, width=width)
        draw.ellipse((cx + int(w * 0.2), base_y - r3, cx + int(w * 0.2) + 2 * r3,
                      base_y + r3), outline=stroke, width=width)
        draw.line((cx - int(w * 0.45), base_y + r1, cx + int(w * 0.55), base_y + r1),
                  fill=stroke, width=width)

    def draw_rain(cx, cy, w):
        start_y = cy + int(12 * scale)
        for offset in (-10, 0, 10):
            x0 = cx + int(offset * scale)
            draw.line((x0, start_y, x0 - int(4 * scale), start_y + int(12 * scale)),
                      fill=stroke, width=width)

    def draw_snow(cx, cy):
        start_y = cy + int(12 * scale)
        for offset in (-10, 0, 10):
            x0 = cx + int(offset * scale)
            draw.line((x0 - int(3 * scale), start_y - int(3 * scale),
                       x0 + int(3 * scale), start_y + int(3 * scale)),
                      fill=stroke, width=width)
            draw.line((x0 - int(3 * scale), start_y + int(3 * scale),
                       x0 + int(3 * scale), start_y - int(3 * scale)),
                      fill=stroke, width=width)

    def draw_lightning(cx, cy):
        pts = [
            (cx - int(4 * scale), cy + int(2 * scale)),
            (cx + int(1 * scale), cy + int(2 * scale)),
            (cx - int(2 * scale), cy + int(14 * scale)),
            (cx + int(6 * scale), cy + int(14 * scale)),
            (cx - int(2 * scale), cy + int(28 * scale)),
        ]
        draw.line(pts, fill=stroke, width=width)

    def draw_fog(cx, cy, w):
        for idx in range(3):
            y0 = cy + int(idx * 6 * scale)
            draw.line((cx - int(w * 0.4), y0, cx + int(w * 0.4), y0),
                      fill=stroke, width=width)

    bucket = meteo_bucket(code)
    is_day = bool(is_day) if is_day is not None else True
    center_x = sx(20)
    center_y = sy(18)
    cloud_w = int(34 * scale)
    cloud_h = int(20 * scale)

    if bucket == "clear":
        if is_day:
            draw_sun(center_x, center_y, int(8 * scale))
        else:
            draw_moon(center_x, center_y, int(8 * scale))
    elif bucket == "partly":
        if is_day:
            draw_sun(sx(14), sy(12), int(7 * scale))
        else:
            draw_moon(sx(14), sy(12), int(7 * scale))
        draw_cloud(sx(20), sy(16), cloud_w, cloud_h)
    elif bucket == "fog":
        draw_fog(sx(20), sy(16), cloud_w)
    elif bucket == "drizzle":
        draw_cloud(sx(20), sy(14), cloud_w, cloud_h)
        draw_rain(sx(20), sy(24), cloud_w)
    elif bucket == "rain":
        draw_cloud(sx(20), sy(14), cloud_w, cloud_h)
        draw_rain(sx(20), sy(24), cloud_w)
    elif bucket == "snow":
        draw_cloud(sx(20), sy(14), cloud_w, cloud_h)
        draw_snow(sx(20), sy(24))
    elif bucket == "thunder":
        draw_cloud(sx(20), sy(14), cloud_w, cloud_h)
        draw_lightning(sx(20), sy(20))
    else:
        draw_cloud(sx(20), sy(14), cloud_w, cloud_h)


def _draw_card(d, x, y, w, h, scale=1.0, fill=245, outline=0, shadow=232):
    """Kachel-Container mit dezentem Schatten."""
    radius = _s(8, scale)
    outline_w = max(1, int(round(scale)))
    shadow_offset = max(1, int(round(2 * scale)))
    if shadow is not None:
        d.rounded_rectangle(
            (x + shadow_offset, y + shadow_offset, x + w + shadow_offset, y + h + shadow_offset),
            radius=radius,
            fill=shadow,
            outline=None,
        )
    d.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=outline_w)


def draw_weather_box(d, img, x, y, w, h, fonts, hourly_map, sun_today_h=None, sun_tomorrow_h=None, scale=1.0):
    _draw_card(d, x, y, w, h, scale=scale)
    icon_size = _s(40, scale)
    icon_x = x + _s(10, scale)
    icon_y = y + int((h - icon_size) / 2)

    now = dt.datetime.now(LOCAL_TZ)
    hour = now.replace(minute=0, second=0, microsecond=0)
    code, is_day = hourly_map.get(hour, (None, None))
    icon = None
    if code is not None:
        try:
            icon = _get_weather_icon_image(
                meteo_bucket(code),
                is_day,
                invert=ICON_INVERT,
                bitreverse=ICON_BITREVERSE,
            )
        except Exception as e:
            logging.warning("Weather-Icon laden fehlgeschlagen: %s", e)
    if icon is not None:
        icon = icon.resize((icon_size, icon_size), resample=Image.NEAREST)
        icon_l = icon.convert("L")
        mask = ImageChops.invert(icon_l)
        img.paste(0, (icon_x, icon_y), mask)
    elif code is not None:
        draw_weather_icon(d, icon_x, icon_y, icon_size, code, is_day, fill=0)
    else:
        draw_weather_icon(d, icon_x, icon_y, icon_size, 3, True, fill=0)

    title = "Wetter"
    title_w, title_h = _text_size(d, title, fonts['title'])
    text_x = x + _s(60, scale)
    lines = []
    if ECO_DEBUG and code is not None:
        lines.append(f"Code: {code}")

    lines.append(f"Sonne heute: {_fmt_hours(sun_today_h)} h")
    lines.append(f"Sonne morgen: {_fmt_hours(sun_tomorrow_h)} h")

    line_heights = [title_h] + [_text_size(d, line, fonts['body'])[1] for line in lines]
    total_h = sum(line_heights) + _s(6, scale)
    start_y = y + max(_s(6, scale), int((h - total_h) / 2))
    _text(d, text_x, start_y, title, font=fonts['title'], fill=0)
    line_y = start_y + title_h + _s(4, scale)
    for line in lines:
        _text(d, text_x, line_y, line, font=fonts['body'], fill=0)
        line_y += _text_size(d, line, fonts['body'])[1] + _s(2, scale)

def minutes_to_hhmm(m):
    if m is None:
        return "-"
    try:
        m = int(m)
        return f"{m//60:02d}:{m%60:02d} h"
    except:
        return "-"


def draw_battery(d, x, y, w, h, soc, arrow=None, fonts=None, scale=1.0):
    soc = max(0, min(100, int(soc) if soc is not None else 0))
    stroke = max(1, int(round(scale)))
    cap_w = _s(6, scale)
    inset = _s(3, scale)
    d.rounded_rectangle((x, y, x + w, y + h), radius=_s(4, scale), outline=0, width=stroke, fill=255)
    d.rectangle((x + w, y + h * 0.35, x + w + cap_w, y + h * 0.65), outline=0, width=stroke)
    inner_w = max(0, int((w - cap_w) * soc / 100))
    d.rectangle((x + inset, y + inset, x + inset + inner_w, y + h - inset), fill=0)
    if fonts:
        _text(d, x + w + _s(12, scale), y + h / 2 - _s(7, scale), f"{soc}%", font=fonts['body'], fill=0)
    # Der frühere Lade-/Entladepfeil wird nicht mehr gezeichnet, da die
    # Richtung aus den Rohdaten nicht zuverlässig ermittelt werden konnte.

def draw_ecoflow_box(d, x, y, w, h, fonts, st, scale=1.0):
    _draw_card(d, x, y, w, h, scale=scale)
    title = "EcoFlow Stream AC"
    title_w, title_h = _text_size(d, title, fonts['title'])
    title_x = x + _s(10, scale)
    title_y = y + _s(6, scale)
    _text(d, title_x, title_y, title, font=fonts['title'], fill=0)

    # Batterie
    batt_x = x + _s(12, scale)
    batt_y = y + int((h - _s(28, scale)) / 2)
    draw_battery(d, batt_x, batt_y, _s(90, scale), _s(28, scale), st.get('soc'), arrow=None, fonts=fonts, scale=scale)

    # Hilfsfunktion
    def fmt_w(v):
        try:
            return f"{int(round(float(v)))} W"
        except Exception:
            return "-"

    # Klarere Zuordnung: Batterie-/Systemleistung, PV-Eingang, Netz, Haushaltslast
    power_w = st.get('power_w') or st.get('gridConnectionPower')
    pv_w    = st.get('pv_input_w_sum')
    grid_w  = st.get('grid_w') or st.get('powGetSysGrid') or st.get('gridConnectionPower')
    load_w  = st.get('load_w') or st.get('powGetSysLoad')

    # Rechenweg: Leistung + PV-Ertrag + Netz = Last
    base_x = x + w - _s(175, scale)  # etwas weiter nach links geschoben
    op_x   = base_x
    lbl_x  = base_x + _s(12, scale)
    val_x  = base_x + _s(105, scale)
    row_height = _s(14, scale)

    entries = [
        ("", "Batterieleistung", power_w),  # Batterie (+ Entladen, - Laden)
        ("+", "PV-Ertrag", pv_w),           # Aktuelle PV-Einspeisung
        ("+", "Netz", grid_w)               # Bezug (+) / Einspeisung (-)
    ]
    block_h = len(entries) * row_height + 14
    base_y = y + max(_s(4, scale), int((h - block_h) / 2))

    for i, (op, label, value) in enumerate(entries):
        y_row = base_y + i * row_height
        if op:
            _text(d, op_x, y_row, op, font=fonts['tiny'], fill=0)
        _text(d, lbl_x, y_row, f"{label}:", font=fonts['tiny'], fill=0)
        _text(d, val_x, y_row, fmt_w(value), font=fonts['tiny'], fill=0)

    # Trennlinie vor dem Ergebnis
    line_y = base_y + len(entries) * row_height + _s(3, scale)
    d.line((base_x, line_y, base_x + _s(120, scale), line_y), fill=0, width=max(1, int(round(scale))))

    result_y = line_y + _s(4, scale)
    _text(d, op_x, result_y, "=", font=fonts['tiny'], fill=0)
    _text(d, lbl_x, result_y, "Last:", font=fonts['tiny'], fill=0)
    _text(d, val_x, result_y, fmt_w(load_w), font=fonts['tiny'], fill=0)
    micro_online = st.get("micro_online")
    if micro_online is not None:
        micro_label = f"Micro online={1 if micro_online else 0}"
        _text(d, x + _s(10, scale), y + h - _s(16, scale), micro_label, font=fonts['tiny'], fill=0)


def draw_info_box(d, info, fonts, y, width, scale=1.0):
    x0 = _s(10, scale)
    low_time = info.get('lowest_today_time')
    low_lbl = (f"{info['lowest_today']/100:.2f} ct @ {low_time.strftime('%H:%M')}"
               if low_time else f"{info['lowest_today']/100:.2f} ct")
    items = [
        ("Preis jetzt", f"{info['current_price']/100:.2f} ct"),
        ("Tief heute",  low_lbl),
        ("Hoch heute",  f"{info['highest_today']/100:.2f} ct"),
    ]
    colw = width / len(items)
    for i,(k,v) in enumerate(items):
        label = f"{k}: {v}"
        label_w, label_h = _text_size(d, label, fonts['bold'])
        col_x = x0 + i * colw
        tx = col_x + (colw - label_w) / 2
        ty = y - label_h / 2
        _text(d, tx, ty, label, font=fonts['bold'], fill=0)

def draw_two_day_chart(img, d, left, right, fonts, subtitles, area,
                       pv_left=None, pv_right=None,
                       cons_left=None, cons_right=None,
                       cur_dt=None, cur_price=None, scale=1.0):
    PRICE_MIN_CENT = 5
    PRICE_MAX_CENT = 60

    X0,Y0,X1,Y1 = area
    W,H = X1-X0, Y1-Y0
    PW  = W/2

    tl, vl = slots_to_15min(left)
    tr, vr = slots_to_15min(right)
    if not (vl or vr): return

    vmin, vmax = PRICE_MIN_CENT, PRICE_MAX_CENT
    sy_price = H/(vmax - vmin if vmax>vmin else 1)

    def _price_to_y(val):
        clipped = max(vmin, min(vmax, val))
        return Y1 - (clipped - vmin) * sy_price

    def vmax_power(series):
        if series is None: return 0
        try:
            if len(series) == 0:
                return 0
            val = float(np.nanmax(series))
            return val if math.isfinite(val) else 0
        except: return 0
    pv_left = pv_left or {}
    pv_right = pv_right or {}
    pv_sum_left = pv_left.get("pv_sum")
    pv_sum_right = pv_right.get("pv_sum")

    pv_max = max(
        vmax_power(pv_sum_left),
        vmax_power(pv_sum_right),
    )
    cons_max = max(vmax_power(cons_left), vmax_power(cons_right))
    power_scale_max = max(pv_max, cons_max)
    power_scale_max = max(power_scale_max * 1.2, 1)
    sy_power = H / power_scale_max

    def _series_has_values(series):
        if series is None:
            return False
        try:
            return np.isfinite(series).any()
        except Exception:
            return False

    has_pv = any(_series_has_values(s) for s in (pv_sum_left, pv_sum_right))

    stroke = max(1, int(round(scale)))
    # Preis-Y-Ticks (nur innerhalb)
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        yy = _price_to_y(yv)
        if Y0 < yy < Y1:
            _text(d, X0 - _s(45, scale), yy - _s(7, scale), f"{yv/100:.2f}", font=fonts['tiny'], fill=0)
        yv += step

    def _price_step_points(xs, val_list):
        points = []
        for i, x in enumerate(xs):
            y = _price_to_y(val_list[i])
            if i == 0:
                points.append((x, y))
            else:
                points.append((x, points[-1][1]))
                points.append((x, y))
        return points

    def _series_to_points(series, xs):
        points = []
        for i, x in enumerate(xs):
            if pd.isna(series.iloc[i]):
                points.append(None)
                continue
            val = max(0.0, min(float(series.iloc[i]), 600.0))
            y = Y1 - val * sy_power
            points.append((x, y))
        return points

    def _segments_from_points(points):
        segment = []
        for pt in points:
            if pt is None:
                if len(segment) > 1:
                    yield segment
                segment = []
                continue
            segment.append(pt)
        if len(segment) > 1:
            yield segment

    def _densify_points(points, steps=3):
        if len(points) < 2:
            return points
        dense = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dense.append((x1, y1))
            for s in range(1, steps + 1):
                t = s / (steps + 1)
                dense.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
        dense.append(points[-1])
        return dense

    def _draw_mask_from_points(mask_draw, series_points):
        for segment in _segments_from_points(series_points):
            smooth = _densify_points(segment, steps=4)
            polygon = smooth + [(smooth[-1][0], Y1), (smooth[0][0], Y1)]
            mask_draw.polygon(polygon, fill=255)

    def _fill_preserve_foreground(shade, mask, base_mask):
        if mask is None:
            return
        safe_mask = ImageChops.multiply(mask, base_mask)
        img.paste(shade, (0, 0), safe_mask)

    price_fill = 232
    pv_fill = 205
    cons_fill = 165
    overlap_fill = 185

    def panel(ts_list, val_list, pv_sum_list, cons_list, x0):
        n = len(ts_list)
        if n < 2: return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        pv_points = None
        cons_points = None
        if has_pv and pv_sum_list is not None and n == len(pv_sum_list):
            pv_points = _series_to_points(_smooth_series(pv_sum_list), xs)
        if cons_list is not None and n == len(cons_list):
            cons_points = _series_to_points(_smooth_series(cons_list), xs)
        if pv_points or cons_points:
            mask_pv = Image.new("L", img.size, 0)
            mask_cons = Image.new("L", img.size, 0)
            if pv_points:
                _draw_mask_from_points(ImageDraw.Draw(mask_pv), pv_points)
            if cons_points:
                _draw_mask_from_points(ImageDraw.Draw(mask_cons), cons_points)
            base_fill_mask = img.point(lambda p: 255 if p >= 250 else 0)
            if cons_points:
                cons_only = ImageChops.subtract(mask_cons, mask_pv)
                _fill_preserve_foreground(cons_fill, cons_only, base_fill_mask)
            if pv_points:
                pv_only = ImageChops.subtract(mask_pv, mask_cons)
                _fill_preserve_foreground(pv_fill, pv_only, base_fill_mask)
            if pv_points and cons_points:
                overlap = ImageChops.multiply(mask_pv, mask_cons)
                _fill_preserve_foreground(overlap_fill, overlap, base_fill_mask)
        step_points = _price_step_points(xs, val_list)
        if step_points:
            price_mask = Image.new("L", img.size, 0)
            price_draw = ImageDraw.Draw(price_mask)
            polygon = step_points + [(step_points[-1][0], Y1), (step_points[0][0], Y1)]
            price_draw.polygon(polygon, fill=255)
            img.paste(price_fill, (0, 0), price_mask)
        # Preis Stufenlinie
        for i in range(n-1):
            x1, y1 = xs[i],   _price_to_y(val_list[i])
            x2, y2 = xs[i+1], _price_to_y(val_list[i+1])
            d.line((x1,y1, x2,y1), fill=0, width=stroke)
            d.line((x2,y1, x2,y2), fill=0, width=stroke)
        # Min/Max Labels
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], _price_to_y(val_list[idx])
            _text(d, xi - _s(12, scale), yi - _s(12, scale), f"{val_list[idx]/100:.2f}", font=fonts['tiny'], fill=0)

    panel(tl, vl, pv_sum_left, cons_left, X0)
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=stroke)
    panel(tr, vr, pv_sum_right, cons_right, X0+PW)

    # Subtitles unter Achse
    _text(d, X0 + _s(5, scale), Y1 + _s(28, scale), subtitles[0], font=fonts['bold'], fill=0)
    _text(d, X0 + PW + _s(5, scale), Y1 + _s(28, scale), subtitles[1], font=fonts['bold'], fill=0)

    # Stundenbeschriftung
    def hour_ticks(ts_list, x0):
        if len(ts_list) < 2: return
        n = len(ts_list)
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        for i,t in enumerate(ts_list):
            if t.minute == 0:
                d.line((xs[i], Y1, xs[i], Y1 + _s(4, scale)), fill=0, width=max(1, int(round(scale))))
                _text(d, xs[i] - _s(8, scale), Y1 + _s(6, scale), t.strftime("%H"), font=fonts['tiny'], fill=0)
    hour_ticks(tl, X0)
    hour_ticks(tr, X0+PW)

    # Legende Leistung
    legend_y = Y0 - _s(18, scale)
    legend_x = X1 - _s(320, scale)
    cursor = legend_x
    for label, shade in (("PV", pv_fill), ("Verbrauch", cons_fill), ("PV>Verbrauch", overlap_fill)):
        _text(d, cursor, legend_y, label, font=fonts['tiny'], fill=0)
        label_w, _ = _text_size(d, label, fonts['tiny'])
        box_x = cursor + label_w + _s(4, scale)
        img.paste(shade, (box_x, legend_y + _s(2, scale), box_x + _s(12, scale), legend_y + _s(12, scale)))
        cursor = box_x + _s(18, scale)
    _text(d, cursor, legend_y, "Preis", font=fonts['tiny'], fill=0)
    if not has_pv:
        _text(d, X0 + _s(6, scale), Y0 + _s(6, scale), "PV DB leer - keine PV-Linien", font=fonts['tiny'], fill=0)

    # Minutengenauer Marker (horizontale Interpolation)
    if cur_price is not None:
        now_dt = dt.datetime.now(LOCAL_TZ)
        marker_dt = now_dt
        if cur_dt is not None and cur_dt > now_dt:
            marker_dt = cur_dt

        def pick_panel_for_marker():
            if len(tl) > 1 and tl[0].date() == marker_dt.date():
                return tl, X0
            if len(tr) > 1 and tr[0].date() == marker_dt.date():
                return tr, X0 + PW
            if len(tl) > 1 and tl[0] <= marker_dt <= tl[-1]:
                return tl, X0
            if len(tr) > 1 and tr[0] <= marker_dt <= tr[-1]:
                return tr, X0 + PW
            return tr, X0 + PW

        arr, x0_panel = pick_panel_for_marker()
        if arr is not None:
            n = len(arr)
            if n > 1:
                t0 = arr[0]
                i_float = (marker_dt - t0).total_seconds() / 900.0  # 900s = 15 min
                i_float = max(0.0, min(n - 1, i_float))
                slot_w = PW / (n - 1)
                px = x0_panel + i_float * slot_w
                py = _price_to_y(cur_price)
                draw_dashed_line(d, px, py, px, Y1 + _s(4, scale), dash=_s(2, scale), gap=_s(3, scale), fill=0, width=max(1, int(round(scale))))
                r = _s(6, scale)
                d.ellipse((px - r, py - r, px + r, py + r), fill=0)
                label = f"{cur_price/100:.2f} ct  {marker_dt.strftime('%H:%M')}"
                tx = px + r + _s(4, scale)
                ty = py - r - _s(10, scale)
                tx = min(tx, X1 - _s(160, scale))
                ty = max(ty, Y0 - _s(18, scale))
                _text(d, tx, ty, label, font=fonts['tiny'], fill=0)

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height
    scale = RENDER_SCALE

    # Daten laden, robust gegen API-Ausfall
    tibber_source = "api"
    try:
        pi = tibber_priceinfo()
        update_price_cache(pi)
    except Exception as e:
        tibber_source = "cache"
        logging.error("Nutze Cache wegen Fehler: %s", e)
        today_cache = load_cache(CACHE_TODAY) or {"data": []}
        yesterday_cache = load_cache(CACHE_YESTERDAY) or {"data": []}
        pi = {
            'today': today_cache.get('data', []),
            'tomorrow': [],
            'current': {'startsAt': dt.datetime.now(LOCAL_TZ).isoformat(),
                        'total': (today_cache.get('data',[{'total':0}])[0].get('total', 0) or 0)}
        }
        logging.info(
            "Tibber Preisinfo aus Cache: heute=%d (Datei), morgen=0, current=%s",
            len(pi.get('today', []) or []),
            pi.get('current', {}).get('startsAt', '-')
        )

    quarter_range = None
    try:
        quarter_range = tibber_priceinfo_quarter_range()
    except Exception as e:
        logging.error("15-Minuten-Preise konnten nicht geladen werden: %s", e)

    tomorrow = pi.get("tomorrow", [])
    if tomorrow:
        left, right = pi["today"], tomorrow
        labels = ("Heute", "Morgen")
        left_date = dt.date.today()
        right_date = dt.date.today() + dt.timedelta(days=1)
    else:
        left, right = (load_cache(CACHE_YESTERDAY) or {"data": []})["data"], pi["today"]
        labels = ("Gestern", "Heute")
        left_date = dt.date.today() - dt.timedelta(days=1)
        right_date = dt.date.today()

    logging.info(
        "Preis-Slots Quelle: %s, linke Achse=%s (%d Werte), rechte Achse=%s (%d Werte)",
        tibber_source,
        labels[0], len(left or []),
        labels[1], len(right or [])
    )

    if quarter_range and (quarter_range.get("today") or quarter_range.get("tomorrow")):
        combined = (quarter_range.get("today") or []) + (quarter_range.get("tomorrow") or [])
        left_q = _filter_range_for_date(combined, left_date)
        right_q = _filter_range_for_date(combined, right_date)
        if left_q:
            left = left_q
        if right_q:
            right = right_q
        logging.info(
            "15-Minuten-Preise via Tibber: today=%d tomorrow=%d current=%s home=%s",
            len(quarter_range.get("today") or []),
            len(quarter_range.get("tomorrow") or []),
            safe_get(quarter_range.get("current") or {}, "startsAt", default="-"),
            quarter_range.get("home_id") or "-"
        )

    left = normalize_price_slots_15min(left)
    right = normalize_price_slots_15min(right)

    today_slots = left if labels[0] == "Heute" else right
    current_price = pick_current_price(quarter_range, pi)
    info = prepare_info(today_slots, current_price)
    
# EcoFlow-Status früh laden (robust) – damit eco immer definiert ist
    eco = {}
    try:
        eco = ecoflow_status_bkw()  # liefert dict mit u.a. pv_input_w_sum, powGetPvSum, powGetSysGrid, powGetSysLoad, power_w, cmsBattSoc
        if ECO_DEBUG:
            logging.info(
                "EcoFlow Live-Status: PV=%s W, Grid=%s W, Load=%s W, SoC=%s%%",
                eco.get('pv_input_w_sum') or eco.get('powGetPvSum'),
                eco.get('grid_w'),
                eco.get('load_w'),
                eco.get('soc')
            )
    except Exception as e:
        logging.error(f"EcoFlow Status fehlgeschlagen: {e}")
        eco = {}

    tl_dt, _ = slots_to_15min(left)
    tr_dt, _ = slots_to_15min(right)

    pv_db_stats(tl_dt, labels[0])
    pv_db_stats(tr_dt, labels[1])

    try:
        hourly = tibber_hourly_consumption(last=48)
    except Exception as e:
        logging.error("Tibber Verbrauchsdaten fehlgeschlagen: %s", e)
        hourly = []
    if hourly:
        logging.info(
            "Tibber Verbrauchsdaten via API: letzte %d Stunden, Start=%s, Ende=%s",
            len(hourly),
            hourly[0][0].strftime("%Y-%m-%d %H:%M"),
            hourly[-1][0].strftime("%Y-%m-%d %H:%M")
        )
    else:
        logging.info("Keine Tibber-Verbrauchsdaten erhalten")
    cons_left = upsample_hourly_to_quarter(tl_dt, hourly)
    cons_right = upsample_hourly_to_quarter(tr_dt, hourly)

    hourly_map = fetch_openmeteo_hourly(api_key.LAT, api_key.LON)
    sun_today_h, sun_tomorrow_h = fetch_openmeteo_sunshine_hours(api_key.LAT, api_key.LON)
    logging.info(
        "Wetterdaten via Open-Meteo (lat=%.4f, lon=%.4f): hourly entries=%d",
        api_key.LAT, api_key.LON, len(hourly_map)
    )
    if sun_today_h is not None or sun_tomorrow_h is not None:
        logging.info(
            "Sonnenstunden: heute=%s h, morgen=%s h",
            _fmt_hours(sun_today_h),
            _fmt_hours(sun_tomorrow_h),
        )
    global SUN_TODAY, SUN_TOMORROW
    SUN_TODAY = sun_today_h
    SUN_TOMORROW = sun_tomorrow_h

    pv_left = get_pv_series_db(tl_dt)
    if labels[1] == "Morgen":
        pv_sum_forecast = pv_forecast_series_for_date(right_date, tr_dt)
        pv_right = {
            "pv1": pd.Series([np.nan] * len(tr_dt), index=tr_dt),
            "pv2": pd.Series([np.nan] * len(tr_dt), index=tr_dt),
            "pv_sum": pv_sum_forecast,
        }
    else:
        pv_right = get_pv_series_db(tr_dt)

    # Canvas (hi-res in L)
    img_hi = Image.new("L", (w * scale, h * scale), 255)
    d_hi = ImageDraw.Draw(img_hi)

    # Fonts (hi-res)
    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", _s(18, scale))
        f_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", _s(14, scale))
        f_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _s(14, scale))
        f_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _s(11, scale))
    except Exception:
        f_title = f_bold = f_body = f_tiny = ImageFont.load_default()
    fonts = {'title': f_title, 'bold': f_bold, 'body': f_body, 'tiny': f_tiny}

    # Layout
    margin = 10
    header_h = 34
    card_h = 78
    box_w = (w - margin * 3) / 2

    # Header bar
    header_y = _s(header_h, scale)
    header_text = "TRMNL Energy"
    update_ts = dt.datetime.now(LOCAL_TZ).strftime("%H:%M %d.%m.%Y")
    _text(d_hi, _s(margin, scale), _s(8, scale), header_text, font=fonts['title'], fill=0)
    ts_w, _ = _text_size(d_hi, update_ts, fonts['tiny'])
    _text(d_hi, img_hi.width - _s(margin, scale) - ts_w, _s(12, scale), update_ts, font=fonts['tiny'], fill=0)
    d_hi.line((0, header_y, img_hi.width, header_y), fill=0, width=max(1, int(round(scale))))

    # Cards
    card_y = header_h + margin
    draw_weather_box(
        d_hi,
        img_hi,
        _s(margin, scale),
        _s(card_y, scale),
        _s(box_w, scale),
        _s(card_h, scale),
        fonts,
        hourly_map,
        sun_today_h=sun_today_h,
        sun_tomorrow_h=sun_tomorrow_h,
        scale=scale,
    )
    draw_ecoflow_box(
        d_hi,
        _s(margin * 2 + box_w, scale),
        _s(card_y, scale),
        _s(box_w, scale),
        _s(card_h, scale),
        fonts,
        eco,
        scale=scale,
    )

    # Info row
    info_y = card_y + card_h + 14
    draw_info_box(d_hi, info, fonts, y=_s(info_y, scale), width=_s(w - 20, scale), scale=scale)

    # Chart area
    chart_top = info_y + 28
    chart_area = (_s(margin, scale), _s(chart_top, scale), _s(w - margin, scale), _s(h - 22, scale))

    draw_two_day_chart(
        img_hi, d_hi, left, right, fonts, labels, chart_area,
        pv_left=pv_left, pv_right=pv_right,
        cons_left=cons_left, cons_right=cons_right,
        cur_dt=info['current_dt'], cur_price=info['current_price'],
        scale=scale,
    )

    img_lo = img_hi.resize((w, h), Image.LANCZOS)
    final_img = render_to_epd(img_lo)

    epd.display(epd.getbuffer(final_img))
    epd.sleep()


if __name__ == "__main__":
    main()

# Akzeptanztests:
# 1) sqlite3 pv_data.db "select count(*) from pv_log;"
#    -> Chart zeichnet 3 PV-Linien (wenn Werte >0 vorhanden)
# 2) DB leer:
#    -> Keine PV-Linie, kein Crash, keine flache 0-Linie bis Mitternacht
# 3) Verbrauch:
#    -> Tibber consumption läuft weiterhin (keine 'NoneType' subscriptable).
