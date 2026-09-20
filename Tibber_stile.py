#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math, json, requests, datetime as dt, sqlite3, logging
from PIL import Image, ImageDraw, ImageFont, ImageChops
import pandas as pd, numpy as np
from urllib.parse import urlencode
import re
from concurrent.futures import ThreadPoolExecutor
from energy_chart_utils import (
    QUARTER_SLOTS_PER_DAY,
    consumption_to_quarter_series,
    format_power_peak,
    merge_consumption_series,
    price_slots_to_quarters,
    quarter_slot_index,
    x_for_quarter_slot,
)
from tibber_live import (
    DEFAULT_DB as TIBBER_SNAPSHOT_DB,
    get_tibber_live_snapshot,
    load_local_quarter_series,
    store_snapshot,
)

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

# Weather icon config
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Image2Lcd exports used by the weather dashboard live beside this file.
WEATHER_ICON_DIR = PROJECT_DIR
ICON_INVERT = True
ICON_BITREVERSE = False
_C_BITMAP_CACHE = {}
_C_IMAGE_CACHE = {}
_BIT_REVERSE_TABLE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))

WEATHER_ICON_FILES = {
    "clear_day": "klar_tag_new.c",
    "clear_night": "klar_nacht_new.c",
    "partly": "wolkig_new.c",
    "overcast": "bewoelkt_new.c",
    "fog": "nebel_new.c",
    "rain": "regen_new.c",
    "showers": "schauer_new.c",
    "thunder": "gewitter_new.c",
    "snow": "schnee_new.c",
}

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

def load_c_bitmap(path, varname=None):
    """Read an Image2Lcd C array and return its payload and dimensions.

    The six-byte Image2Lcd header is ``00 01 width-le16 height-le16``.  Rows
    are MSB-first, one bit per pixel, and padded to a complete byte.  The
    declaration is parsed instead of deriving the array name from the file.
    """
    if path in _C_BITMAP_CACHE:
        return _C_BITMAP_CACHE[path]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Icon-Datei nicht lesbar: {path}: {e}")
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    content = re.sub(r"//.*", "", content)
    declaration = re.search(
        r"(?:const\s+)?unsigned\s+char\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\}",
        content, re.S,
    )
    if not declaration:
        raise RuntimeError(f"Kein unsigned-char-Array gefunden in {path}")
    array_name, declared_length, array_body = declaration.groups()
    if varname is not None and array_name != varname:
        raise RuntimeError(f"Array {varname} nicht gefunden in {path}")
    bytes_list = [int(b, 16) for b in re.findall(r"0[xX]([0-9A-Fa-f]{2})", array_body)]
    if not bytes_list:
        raise RuntimeError(f"Keine Icon-Daten gefunden in {path}")
    if len(bytes_list) != int(declared_length):
        raise RuntimeError(f"Arraylänge in {path} ist inkonsistent")
    if len(bytes_list) < 6 or bytes_list[:2] != [0, 1]:
        raise RuntimeError(f"Unbekannter Image2Lcd-Header in {path}")
    width = bytes_list[2] | bytes_list[3] << 8
    height = bytes_list[4] | bytes_list[5] << 8
    data = bytes(bytes_list[6:])
    if len(data) != ((width + 7) // 8) * height:
        raise RuntimeError(f"Ungültige Bitmap-Datenlänge in {path}")
    result = (data, width, height, array_name)
    _C_BITMAP_CACHE[path] = result
    return result

def c_bitmap_to_image(data, w, h, invert=False, bitreverse=False):
    if bitreverse:
        data = data.translate(_BIT_REVERSE_TABLE)
    img = Image.frombytes("1", (w, h), data)
    if invert:
        img = ImageChops.invert(img)
    return img

def get_weather_icon_filename(code, is_day):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return WEATHER_ICON_FILES["overcast"]
    if code == 0:
        return WEATHER_ICON_FILES["clear_day" if is_day is not False else "clear_night"]
    if code in (1, 2):
        return WEATHER_ICON_FILES["partly"]
    if code == 3:
        return WEATHER_ICON_FILES["overcast"]
    if code in (45, 48):
        return WEATHER_ICON_FILES["fog"]
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67):
        return WEATHER_ICON_FILES["rain"]
    if code in (71, 73, 75, 77, 85, 86):
        return WEATHER_ICON_FILES["snow"]
    if code in (80, 81, 82):
        return WEATHER_ICON_FILES["showers"]
    if code in (95, 96, 99):
        return WEATHER_ICON_FILES["thunder"]
    return WEATHER_ICON_FILES["overcast"]


def _get_weather_icon_image(code, is_day, invert=False, bitreverse=False):
    filename = get_weather_icon_filename(code, is_day)
    if not filename:
        return None
    path = os.path.join(WEATHER_ICON_DIR, filename)
    data, w, h, _array_name = load_c_bitmap(path)
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
def slots_to_15min(slots):
    """Normalize hourly or quarter-hourly Tibber prices to 15-minute slots."""
    return price_slots_to_quarters(slots, LOCAL_TZ)

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

# ---------- Tibber Consumption ----------
def _tibber_consumption_request(resolution, last):
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    q = f"""
    {{ viewer {{ homes {{
      consumption(resolution: {resolution}, last: {last}) {{
        nodes {{ from consumption }}
      }}
    }}}} }}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": q}, headers=hdr, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(j["errors"][0].get("message", "Tibber GraphQL error"))
    homes = (((j.get("data") or {}).get("viewer") or {}).get("homes") or [])
    home = pick_home_with_data(homes) or {}
    cons = (home.get("consumption") or {})
    nodes = (cons.get("nodes") or []) if isinstance(cons, dict) else []
    if not nodes:
        logging.info("Tibber Consumption leer/fehlend: homes=%d", len(homes))
        return []
    return nodes


def tibber_consumption():
    """Fetch Tibber's historical consumption at its supported hourly resolution."""
    return {"resolution": "hourly", "nodes": _tibber_consumption_request("HOURLY", 48)}


def fetch_and_store_live_snapshot():
    """Best-effort one-shot Pulse read; never makes display rendering fatal."""
    try:
        preferred_home_id = os.getenv("TIBBER_LIVE_HOME_ID") or None
        snapshot = get_tibber_live_snapshot(api_key.API_KEY, preferred_home_id)
        inserted, interval = store_snapshot(snapshot, TIBBER_SNAPSHOT_DB)
        logging.info("Tibber Pulse snapshot: timestamp=%s power=%sW inserted=%s interval=%s",
                     snapshot.timestamp.isoformat(), snapshot.power_w, inserted,
                     (interval or {}).get("quality", "first_snapshot"))
        return snapshot
    except Exception as exc:
        logging.warning("Tibber Pulse snapshot unavailable; continuing without live data: %s", exc)
        return None

# ---------- Wetter ----------
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_MODELS = (
    ("icon_d2", "ICON-D2"),
    ("icon_eu", "ICON-EU (fallback)"),
)


def _required_weather_times(today):
    """Return every hourly timestamp needed by the two dashboard days."""
    start = dt.datetime.combine(today, dt.time(6), tzinfo=LOCAL_TZ)
    end = dt.datetime.combine(today + dt.timedelta(days=2), dt.time(6),
                              tzinfo=LOCAL_TZ)
    required = set()
    current = start
    while current < end:
        required.add(current)
        current += dt.timedelta(hours=1)
    return required


def _parse_openmeteo_response(payload):
    hourly = payload.get("hourly", {}) or {}
    columns = (
        hourly.get("time") or [],
        hourly.get("temperature_2m") or [],
        hourly.get("precipitation_probability") or [],
        hourly.get("weather_code") or hourly.get("weathercode") or [],
        hourly.get("is_day") or [],
    )
    hourly_map = {}
    for t_str, temperature, rain, code, is_day in zip(*columns):
        try:
            timestamp = dt.datetime.fromisoformat(t_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=LOCAL_TZ)
            else:
                timestamp = timestamp.astimezone(LOCAL_TZ)
            hourly_map[timestamp] = {
                "temperature": _as_float_or_none(temperature),
                "precipitation_probability": _as_float_or_none(rain),
                "code": int(code),
                "is_day": bool(is_day),
            }
        except (TypeError, ValueError):
            continue

    daily = payload.get("daily", {}) or {}
    sunshine_by_date = {}
    for date_string, seconds in zip(
            daily.get("time") or [], daily.get("sunshine_duration") or []):
        try:
            value = _as_float_or_none(seconds)
            sunshine_by_date[dt.date.fromisoformat(date_string)] = (
                value / 3600.0 if value is not None else None
            )
        except (TypeError, ValueError):
            continue
    return hourly_map, sunshine_by_date


def _fetch_openmeteo_model(lat, lon, model, today):
    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,weather_code,is_day",
            "daily": "sunshine_duration",
            "forecast_days": 3,
            "timezone": "Europe/Berlin",
            "models": model,
        },
        timeout=10,
    )
    response.raise_for_status()
    hourly_map, sunshine_by_date = _parse_openmeteo_response(response.json())
    required = _required_weather_times(today)
    missing = required.difference(hourly_map)
    incomplete = {
        timestamp for timestamp in required.intersection(hourly_map)
        if any(hourly_map[timestamp].get(field) is None
               for field in ("temperature", "precipitation_probability", "code"))
    }
    sunshine = tuple(sunshine_by_date.get(today + dt.timedelta(days=offset))
                     for offset in (0, 1))
    if missing or incomplete or any(value is None for value in sunshine):
        raise ValueError(
            "%s forecast incomplete: %d hourly values missing, %d incomplete, "
            "and %d sunshine totals missing"
            % (model, len(missing), len(incomplete),
               sum(value is None for value in sunshine))
        )
    return hourly_map, sunshine


def fetch_openmeteo_forecast(lat, lon, include_model=False):
    """
    Fetch the three-day hourly forecast and two daily sunshine totals.

    Three forecast days are intentional: tomorrow's night ends at 06:00 on
    the day after tomorrow.
    """
    today = dt.datetime.now(LOCAL_TZ).date()
    for model, display_name in WEATHER_MODELS:
        try:
            hourly_map, sunshine = _fetch_openmeteo_model(lat, lon, model, today)
            result = (hourly_map, sunshine, display_name)
            return result if include_model else result[:2]
        except Exception as error:
            logging.warning("Open-Meteo model %s unavailable: %s", model, error)
    result = ({}, (None, None), None)
    return result if include_model else result[:2]


def fetch_openmeteo_hourly(lat, lon):
    """Compatibility wrapper for callers which only need hourly data."""
    return fetch_openmeteo_forecast(lat, lon)[0]

def fetch_openmeteo_sunshine_hours(lat, lon):
    """
    Holt Open-Meteo daily sunshine_duration (Sekunden) für heute und morgen.
    Return: (sun_today_h, sun_tomorrow_h) in Stunden oder (None, None)
    """
    return fetch_openmeteo_forecast(lat, lon)[1]


WEATHER_PERIODS = (
    ("Vorm.", 6, 12, 0),
    ("Nachm.", 12, 18, 0),
    ("Abend", 18, 22, 0),
    ("Nacht", 22, 6, 1),
)

_WEATHER_SEVERITY = {
    "clear": 0, "partly": 1, "cloudy": 2, "overcast": 3,
    "fog": 4, "drizzle": 5, "rain": 6, "snow": 7, "thunder": 9,
}


def weather_code_severity(code):
    """Rank meaningful WMO events above merely frequent benign conditions."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return 2
    if code in (95, 96, 99):
        return 10
    if code in (65, 67, 82):
        return 9
    if code in (71, 73, 75, 77, 85, 86):
        return 8
    return _WEATHER_SEVERITY.get(meteo_bucket(code), 2)


def aggregate_weather_period(hourly_map, day, start_hour, end_hour, end_day_offset=0):
    """Aggregate one explicit half-open local-time interval."""
    start = dt.datetime.combine(day, dt.time(start_hour), tzinfo=LOCAL_TZ)
    end_day = day + dt.timedelta(days=end_day_offset)
    end = dt.datetime.combine(end_day, dt.time(end_hour), tzinfo=LOCAL_TZ)
    rows = [row for timestamp, row in hourly_map.items() if start <= timestamp < end]
    temperatures = [row.get("temperature") for row in rows
                    if row.get("temperature") is not None]
    rain = [row.get("precipitation_probability") for row in rows
            if row.get("precipitation_probability") is not None]
    representative = max(
        rows,
        key=lambda row: weather_code_severity(row.get("code")),
        default=None,
    )
    return {
        "temperature": round(sum(temperatures) / len(temperatures)) if temperatures else None,
        "precipitation_probability": round(max(rain)) if rain else None,
        "code": representative.get("code") if representative else None,
        "is_day": representative.get("is_day") if representative else start_hour != 22,
        "count": len(rows),
    }


def aggregate_weather_days(hourly_map, today=None):
    today = today or dt.datetime.now(LOCAL_TZ).date()
    result = []
    for day_offset in (0, 1):
        day = today + dt.timedelta(days=day_offset)
        result.append([
            aggregate_weather_period(hourly_map, day, start, end, next_day)
            for _label, start, end, next_day in WEATHER_PERIODS
        ])
    return result

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


def _fmt_hours(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"


_BAYER_4X4 = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]

_BAYER_8X8 = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
]

def _ordered_dither_bayer(image_l, matrix=_BAYER_8X8, strength=0.0, level=None):
    if image_l.mode != "L":
        image_l = image_l.convert("L")
    m = matrix or _BAYER_8X8
    size = len(m)
    strength = max(0.0, min(1.0, float(strength)))
    if level is not None:
        bias = max(0, min(255, int(level)))
    else:
        bias = int(round(90 * strength))
    threshold_scale = 255.0 / (size * size)
    w, h = image_l.size
    out = Image.new("1", (w, h), 1)
    src = image_l.load()
    dst = out.load()
    for y in range(h):
        row = m[y % size]
        for x in range(w):
            v = src[x, y]
            v = max(0, min(255, v - bias))
            threshold = (row[x % size] + 0.5) * threshold_scale
            dst[x, y] = 0 if v < threshold else 255
    return out


def _make_bayer_tile(density, size=8):
    density = max(0.0, min(1.0, float(density)))
    threshold = density * (size * size)
    tile = Image.new("1", (size, size), 1)
    px = tile.load()
    for y in range(size):
        for x in range(size):
            if _BAYER_8X8[y][x] < threshold:
                px[x, y] = 0
    return tile


def _tile_pattern(tile, size):
    pattern = Image.new("1", size, 1)
    tw, th = tile.size
    for y in range(0, size[1], th):
        for x in range(0, size[0], tw):
            pattern.paste(tile, (x, y))
    return pattern


def _paste_dithered_polygon(img, polygon, density=0.3):
    if not polygon:
        return
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x = int(max(min(xs), 0))
    min_y = int(max(min(ys), 0))
    max_x = int(min(max(xs), img.width - 1))
    max_y = int(min(max(ys), img.height - 1))
    if max_x <= min_x or max_y <= min_y:
        return
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    mask = Image.new("1", (bbox_w, bbox_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    offset_polygon = [(x - min_x, y - min_y) for (x, y) in polygon]
    mask_draw.polygon(offset_polygon, fill=1)
    tile = _make_bayer_tile(density)
    pattern = _tile_pattern(tile, (bbox_w, bbox_h))
    img.paste(pattern, (min_x, min_y), mask)


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
    width = 2
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


def _draw_raindrop(draw, x, y):
    draw.polygon(((x + 3, y), (x, y + 6), (x + 1, y + 9),
                  (x + 5, y + 9), (x + 6, y + 6)), outline=0, fill=None)


def draw_weather_dashboard(d, img, x, y, w, h, fonts, weather_days,
                           sun_today_h=None, sun_tomorrow_h=None):
    """Render two equal days, each containing four equal forecast periods."""
    half_w = w / 2
    header_h = 25
    cell_w = half_w / 4
    suns = (sun_today_h, sun_tomorrow_h)
    day_names = ("Wetter heute", "Wetter morgen")
    d.line((x, y + h, x + w, y + h), fill=0, width=1)
    for day_index in range(2):
        day_x = x + day_index * half_w
        d.text((day_x + 5, y + 4), day_names[day_index], font=fonts["bold"], fill=0)
        sunshine = f"Sonne: {_fmt_hours(suns[day_index]).replace('.', ',')} h"
        sunshine_w, _ = _text_size(d, sunshine, fonts["small"])
        d.text((day_x + half_w - sunshine_w - 5, y + 5), sunshine,
               font=fonts["small"], fill=0)
        d.line((day_x, y + header_h, day_x + half_w, y + header_h), fill=0, width=1)
        for period_index, (label, _start, _end, _offset) in enumerate(WEATHER_PERIODS):
            cell_x = day_x + period_index * cell_w
            data = weather_days[day_index][period_index]
            label_w, _ = _text_size(d, label, fonts["tiny"])
            d.text((cell_x + (cell_w - label_w) / 2, y + 29), label,
                   font=fonts["tiny"], fill=0)
            temperature = "--°" if data["temperature"] is None else f'{data["temperature"]}°'
            temp_w, _ = _text_size(d, temperature, fonts["temperature"])
            d.text((cell_x + (cell_w - temp_w) / 2, y + 43), temperature,
                   font=fonts["temperature"], fill=0)
            icon_x = int(cell_x + (cell_w - 90) / 2)
            icon_y = y + 60
            icon = None
            if data["code"] is not None:
                try:
                    icon = _get_weather_icon_image(
                        data["code"], data["is_day"],
                        invert=ICON_INVERT, bitreverse=ICON_BITREVERSE,
                    )
                except Exception as exc:
                    logging.warning("Weather-Icon laden fehlgeschlagen: %s", exc)
            if icon is not None:
                img.paste(icon, (icon_x, icon_y))
            else:
                draw_weather_icon(d, icon_x + 21, icon_y + 20, 48,
                                  data["code"] if data["code"] is not None else 3,
                                  data["is_day"], fill=0)
            rain = ("-- %" if data["precipitation_probability"] is None
                    else f'{data["precipitation_probability"]} %')
            rain_w, _ = _text_size(d, rain, fonts["tiny"])
            rain_x = cell_x + (cell_w - rain_w - 10) / 2
            _draw_raindrop(d, int(rain_x), y + h - 17)
            d.text((rain_x + 9, y + h - 19), rain, font=fonts["tiny"], fill=0)
    d.line((x + half_w, y, x + half_w, y + h), fill=0, width=3)

def minutes_to_hhmm(m):
    if m is None:
        return "-"
    try:
        m = int(m)
        return f"{m//60:02d}:{m%60:02d} h"
    except:
        return "-"


def draw_battery(d, x, y, w, h, soc, arrow=None, fonts=None):
    soc = max(0, min(100, int(soc) if soc is not None else 0))
    d.rectangle((x, y, x+w, y+h), outline=0, width=2)
    d.rectangle((x+w, y+h*0.35, x+w+6, y+h*0.65), outline=0, width=2)
    inner_w = max(0, int((w-6) * soc/100))
    d.rectangle((x+3, y+3, x+3+inner_w, y+h-3), fill=0)
    if fonts: d.text((x+w+12, y+h/2-7), f"{soc}%", font=fonts['small'], fill=0)
    # Der frühere Lade-/Entladepfeil wird nicht mehr gezeichnet, da die
    # Richtung aus den Rohdaten nicht zuverlässig ermittelt werden konnte.

def draw_ecoflow_box(d, x, y, w, h, fonts, st):
    d.rectangle((x, y, x + w, y + h), outline=0, width=2)
    title = "EcoFlow Stream AC"
    title_w, title_h = _text_size(d, title, fonts['bold'])
    title_x = x + 10
    title_y = y + 4
    d.text((title_x, title_y), title, font=fonts['bold'], fill=0)

    # Batterie
    batt_x = x + 12
    batt_y = y + int((h - 28) / 2)
    draw_battery(d, batt_x, batt_y, 90, 28, st.get('soc'), arrow=None, fonts=fonts)

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
    base_x = x + w - 175  # etwas weiter nach links geschoben
    op_x   = base_x
    lbl_x  = base_x + 12
    val_x  = base_x + 105
    row_height = 14

    entries = [
        ("", "Batterieleistung", power_w),  # Batterie (+ Entladen, - Laden)
        ("+", "PV-Ertrag", pv_w),           # Aktuelle PV-Einspeisung
        ("+", "Netz", grid_w)               # Bezug (+) / Einspeisung (-)
    ]
    block_h = len(entries) * row_height + 14
    base_y = y + max(4, int((h - block_h) / 2))

    for i, (op, label, value) in enumerate(entries):
        y_row = base_y + i * row_height
        if op:
            d.text((op_x, y_row), op, font=fonts['tiny'], fill=0)
        d.text((lbl_x, y_row), f"{label}:", font=fonts['tiny'], fill=0)
        d.text((val_x, y_row), fmt_w(value), font=fonts['tiny'], fill=0)

    # Trennlinie vor dem Ergebnis
    line_y = base_y + len(entries) * row_height + 3
    d.line((base_x, line_y, base_x + 120, line_y), fill=0, width=1)

    result_y = line_y + 4
    d.text((op_x, result_y), "=", font=fonts['tiny'], fill=0)
    d.text((lbl_x, result_y), "Last:", font=fonts['tiny'], fill=0)
    d.text((val_x, result_y), fmt_w(load_w), font=fonts['tiny'], fill=0)
    micro_online = st.get("micro_online")
    if micro_online is not None:
        micro_label = f"Micro online={1 if micro_online else 0}"
        d.text((x + 10, y + h - 14), micro_label, font=fonts['tiny'], fill=0)


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
        label = f"{k}: {v}"
        label_w, label_h = _text_size(d, label, fonts['bold'])
        col_x = x0 + i * colw
        tx = col_x + (colw - label_w) / 2
        ty = y - label_h / 2
        d.text((tx, ty), label, font=fonts['bold'], fill=0)


def format_live_power(watts):
    if watts is None or not math.isfinite(float(watts)) or float(watts) < 0:
        return "-- W"
    watts = float(watts)
    return f"{watts / 1000:.2f} kW".replace(".", ",") if watts >= 1000 else f"{watts:.0f} W"


def draw_live_consumption_box(d, fonts, snapshot, x, y, w=142, h=62):
    """Draw the optional Pulse readout over the chart without resizing it."""
    d.rectangle((x, y, x + w, y + h), fill=255, outline=0, width=2)
    d.text((x + 7, y + 5), "VERBRAUCH AKTUELL", font=fonts['tiny'], fill=0)
    power = snapshot.power_w if snapshot is not None else None
    d.text((x + 7, y + 20), format_live_power(power), font=fonts['bold'], fill=0)
    last_hour = snapshot.accumulated_consumption_last_hour_kwh if snapshot is not None else None
    if last_hour is not None and math.isfinite(float(last_hour)) and 0 <= float(last_hour) <= 100:
        label = f"Stunde  {float(last_hour):.2f} kWh".replace(".", ",")
        d.text((x + 7, y + 43), label, font=fonts['tiny'], fill=0)


def draw_two_day_chart(img, d, left, right, fonts, subtitles, area,
                       pv_left=None, pv_right=None,
                       cons_left=None, cons_right=None,
                       cur_dt=None, cur_price=None):
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
            finite = [float(value) for value in series
                      if value is not None and math.isfinite(float(value))]
            if not finite:
                return 0
            return max(finite)
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
            return any(value is not None and math.isfinite(float(value)) for value in series)
        except Exception:
            return False

    has_pv = any(_series_has_values(s) for s in (pv_sum_left, pv_sum_right))

    # Preis-Y-Ticks (nur innerhalb)
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        yy = _price_to_y(yv)
        if Y0 < yy < Y1:
            d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['tiny'], fill=0)
        yv += step

    def _draw_price_shadow(xs, val_list):
        shadow_h = 40
        for i in range(len(xs) - 1):
            x_start = int(xs[i])
            x_end = int(xs[i + 1])
            y_base = int(_price_to_y(val_list[i]))
            for offset in range(1, shadow_h + 1):
                y = y_base + offset
                if y >= Y1:
                    break
                density = max(0.0, (shadow_h - offset) / shadow_h)
                step = 6
                threshold = max(1, int(round(density * step)))
                if threshold <= 0:
                    continue
                for x in range(x_start, x_end + 1, 2):
                    if ((x + offset) % step) < threshold:
                        d.point((x, y), fill=0)

    def _series_to_points(series, xs):
        points = []
        for i, x in enumerate(xs):
            value = series.iloc[i] if hasattr(series, "iloc") else series[i]
            if value is None or pd.isna(value):
                points.append(None)
                continue
            val = max(0.0, float(value))
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

    pv_fill_gray = 200
    pv_dither_strength = 0.3

    def panel(ts_list, val_list, pv_sum_list, cons_list, x0, panel_label):
        n = len(ts_list)
        if n < 2: return
        xs = [x_for_quarter_slot(x0, PW, quarter_slot_index(timestamp))
              for timestamp in ts_list]
        pv_points = None
        cons_points = None
        if has_pv and pv_sum_list is not None and n == len(pv_sum_list):
            pv_points = _series_to_points(_smooth_series(pv_sum_list), xs)
        if cons_list is not None and len(cons_list) == QUARTER_SLOTS_PER_DAY:
            cons_xs = [x_for_quarter_slot(x0, PW, slot)
                       for slot in range(QUARTER_SLOTS_PER_DAY)]
            cons_points = _series_to_points(cons_list, cons_xs)
        if pv_points:
            pv_layer = Image.new("L", img.size, 255)
            pv_draw = ImageDraw.Draw(pv_layer)
            for segment in _segments_from_points(pv_points):
                smooth = _densify_points(segment, steps=4)
                polygon = smooth + [(smooth[-1][0], Y1), (smooth[0][0], Y1)]
                pv_draw.polygon(polygon, fill=pv_fill_gray)
            pv_mask = pv_layer.point(lambda p: 255 if p < 255 else 0)
            pv_dither = _ordered_dither_bayer(pv_layer, matrix=_BAYER_8X8, strength=pv_dither_strength)
            img.paste(pv_dither, (0, 0), pv_mask)
        # Verbrauch als gut sichtbare Kurve statt als gefuellte Flaeche.
        if cons_points:
            # Never bridge missing quarters: an hourly value or a Pulse gap
            # remains a measured point, not an invented higher-resolution line.
            for segment in _segments_from_points(cons_points):
                d.line(segment, fill=0, width=3)
            for point in (point for point in cons_points if point is not None):
                d.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=0)
        _draw_price_shadow(xs, val_list)
        # Preis Stufenlinie
        for i in range(n):
            x1, y1 = xs[i],   _price_to_y(val_list[i])
            next_slot = quarter_slot_index(ts_list[i]) + 1
            x2 = x_for_quarter_slot(x0, PW, next_slot)
            d.line((x1,y1, x2,y1), fill=0, width=2)
            if i + 1 < n:
                y2 = _price_to_y(val_list[i+1])
                d.line((x2,y1, x2,y2), fill=0, width=2)
        # Mark the highest measured consumption in each day panel.  The raw
        # value is used, rather than the visually smoothed curve, so the label
        # remains an accurate reading.
        if cons_points and _series_has_values(cons_list):
            peak_index = int(np.nanargmax(np.asarray(cons_list, dtype=float)))
            peak_watts = float(cons_list.iloc[peak_index] if hasattr(cons_list, "iloc")
                               else cons_list[peak_index])
            if cons_points[peak_index] is not None:
                peak_x, peak_y = cons_points[peak_index]
                radius = 5
                peak_label = f"Peak {format_power_peak(peak_watts)}"
                label_w, label_h = _text_size(d, peak_label, fonts['tiny'])
                label_x = max(x0 + 2, min(peak_x - label_w / 2, x0 + PW - label_w - 2))
                label_y = peak_y - label_h - radius - 3
                if label_y < Y0 + 2:
                    label_y = peak_y + radius + 3
                leader_x = max(label_x, min(peak_x, label_x + label_w))
                leader_y = label_y if label_y > peak_y else label_y + label_h
                d.line((peak_x, peak_y, leader_x, leader_y), fill=0, width=1)
                d.rectangle(
                    (label_x - 2, label_y - 1, label_x + label_w + 2, label_y + label_h + 1),
                    fill=255,
                )
                d.text((label_x, label_y), peak_label, font=fonts['tiny'], fill=0)
                d.ellipse(
                    (peak_x - radius, peak_y - radius, peak_x + radius, peak_y + radius),
                    fill=255, outline=0, width=2,
                )
                logging.info(
                    "Peak %s rendering: slot=%d, value=%.1f W, x=%.2f, y=%.2f",
                    panel_label.lower(), peak_index, peak_watts, peak_x, peak_y,
                )
        # Min/Max Labels
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], _price_to_y(val_list[idx])
            d.text((xi-12, yi-12), f"{val_list[idx]/100:.2f}", font=fonts['tiny'], fill=0)

    panel(tl, vl, pv_sum_left, cons_left, X0, subtitles[0])
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)
    panel(tr, vr, pv_sum_right, cons_right, X0+PW, subtitles[1])

    # Subtitles unter Achse
    d.text((X0+5,    Y1+28), subtitles[0], font=fonts['bold'], fill=0)
    d.text((X0+PW+5, Y1+28), subtitles[1], font=fonts['bold'], fill=0)

    # Stundenbeschriftung
    def hour_ticks(ts_list, x0):
        if len(ts_list) < 2: return
        for hour in range(24):
            slot = hour * 4
            x = x_for_quarter_slot(x0, PW, slot)
            d.line((x, Y1, x, Y1+4), fill=0, width=1)
            d.text((x-8, Y1+6), f"{hour:02d}", font=fonts['tiny'], fill=0)
    hour_ticks(tl, X0)
    hour_ticks(tr, X0+PW)

    # Legende Leistung
    legend_y = Y0 - 18
    legend_x = X1 - 320
    cursor = legend_x
    for label, density in (("PV", 0.6),):
        d.text((cursor, legend_y), label, font=fonts['tiny'], fill=0)
        label_w, _ = _text_size(d, label, fonts['tiny'])
        box_x = cursor + label_w + 4
        _paste_dithered_polygon(
            img,
            [(box_x, legend_y + 2), (box_x + 12, legend_y + 2),
             (box_x + 12, legend_y + 12), (box_x, legend_y + 12)],
            density=density,
        )
        cursor = box_x + 18
    label = "Verbrauch"
    d.text((cursor, legend_y), label, font=fonts['tiny'], fill=0)
    label_w, _ = _text_size(d, label, fonts['tiny'])
    line_x = cursor + label_w + 4
    d.line((line_x, legend_y + 7, line_x + 12, legend_y + 7), fill=0, width=3)
    cursor = line_x + 18
    d.text((cursor, legend_y), "Preis", font=fonts['tiny'], fill=0)
    if not has_pv:
        d.text((X0 + 6, Y0 + 6), "PV DB leer - keine PV-Linien", font=fonts['tiny'], fill=0)

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
                current_slot = quarter_slot_index(marker_dt)
                px = x_for_quarter_slot(x0_panel, PW, current_slot)
                py = _price_to_y(cur_price)
                draw_dashed_line(d, px, py, px, Y1 + 4, dash=2, gap=3, fill=0, width=1)
                r = 6
                d.ellipse((px - r, py - r, px + r, py + r), fill=0)
                label = f"{cur_price/100:.2f} ct  {marker_dt.strftime('%H:%M')}"
                tx = px + r + 4
                ty = py - r - 10
                tx = min(tx, X1 - 160)
                ty = max(ty, Y0 - 18)
                d.text((tx, ty), label, font=fonts['tiny'], fill=0)

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height

    def _future_result(fut, default, error_msg):
        try:
            return fut.result()
        except Exception as e:
            logging.error(error_msg, e)
            return default

    # Daten parallel vorab laden (API + DB), Display-Update erst am Ende.
    with ThreadPoolExecutor(max_workers=5) as executor:
        fut_pi = executor.submit(tibber_priceinfo)
        fut_quarter = executor.submit(tibber_priceinfo_quarter_range)
        fut_consumption = executor.submit(tibber_consumption)
        fut_weather = executor.submit(
            fetch_openmeteo_forecast,
            getattr(api_key, "LAT", 0), getattr(api_key, "LON", 0),
            True,
        )
        fut_live = executor.submit(fetch_and_store_live_snapshot)

        # Daten laden, robust gegen API-Ausfall
        tibber_source = "api"
        pi = _future_result(fut_pi, None, "Tibber Preisinfo fehlgeschlagen: %s")
        if pi:
            update_price_cache(pi)
        else:
            tibber_source = "cache"
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

        quarter_range = _future_result(
            fut_quarter, None, "15-Minuten-Preise konnten nicht geladen werden: %s"
        )

        consumption = _future_result(
            fut_consumption,
            {"resolution": "hourly", "nodes": []},
            "Tibber Verbrauchsdaten fehlgeschlagen: %s",
        )
        hourly_map, sunshine, weather_model = _future_result(
            fut_weather, ({}, (None, None), None),
            "Wetterdaten konnten nicht geladen werden: %s"
        )
        sun_today_h, sun_tomorrow_h = sunshine
        live_snapshot = _future_result(
            fut_live, None, "Tibber Pulse snapshot failed unexpectedly: %s"
        )

    # Consumption and prices deliberately share the same two complete day
    # panels.  Tomorrow's prices remain cached by the regular Tibber flow, but
    # this historical chart is always yesterday versus today.
    left = (load_cache(CACHE_YESTERDAY) or {"data": []})["data"]
    right = pi["today"]
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
    logging.info(
        "Preis-Slots normalisiert: %s=%d, %s=%d",
        labels[0].lower(), len(left), labels[1].lower(), len(right),
    )

    today_slots = left if labels[0] == "Heute" else right
    current_price = pick_current_price(quarter_range, pi)
    info = prepare_info(today_slots, current_price)
    
    tl_dt, _ = slots_to_15min(left)
    tr_dt, _ = slots_to_15min(right)

    pv_db_stats(tl_dt, labels[0])
    pv_db_stats(tr_dt, labels[1])
    consumption_nodes = consumption.get("nodes") or []
    consumption_resolution = consumption.get("resolution", "hourly")
    if consumption_nodes:
        parsed_consumption_times = []
        for node in consumption_nodes:
            try:
                parsed_consumption_times.append(
                    dt.datetime.fromisoformat(node["from"]).astimezone(LOCAL_TZ)
                )
            except (KeyError, TypeError, ValueError):
                pass
        logging.info(
            "Tibber Verbrauchsdaten via API: n=%d, resolution=%s, Zeitstempel=%s",
            len(consumption_nodes), consumption_resolution,
            ", ".join(value.isoformat() for value in parsed_consumption_times),
        )
    else:
        logging.info("Keine Tibber-Verbrauchsdaten erhalten")
    cons_left_times, cons_left_values = consumption_to_quarter_series(
        consumption_nodes, left_date, LOCAL_TZ, consumption_resolution
    )
    cons_right_times, cons_right_values = consumption_to_quarter_series(
        consumption_nodes, right_date, LOCAL_TZ, consumption_resolution
    )
    cons_left_values = merge_consumption_series(
        cons_left_values, load_local_quarter_series(left_date, TIBBER_SNAPSHOT_DB)
    )
    cons_right_values = merge_consumption_series(
        cons_right_values, load_local_quarter_series(right_date, TIBBER_SNAPSHOT_DB)
    )
    cons_left = pd.Series(cons_left_values)
    cons_right = pd.Series(cons_right_values)

    def log_consumption_summary(label, timestamps):
        present = [timestamp for timestamp in timestamps if timestamp is not None]
        logging.info(
            "Consumption %s: n=%d, resolution=%s, first_timestamp=%s, last_timestamp=%s",
            label.lower(), len(present), consumption_resolution,
            present[0].isoformat() if present else "-",
            present[-1].isoformat() if present else "-",
        )
    log_consumption_summary(labels[0], cons_left_times)
    log_consumption_summary(labels[1], cons_right_times)

    def log_consumption_peak(label, timestamps, values):
        valid = [(index, value) for index, value in enumerate(values)
                 if value is not None and math.isfinite(float(value))]
        if not valid:
            logging.info("Peak %s: keine gültigen Verbrauchswerte", label.lower())
            return
        peak_index, peak_value = max(valid, key=lambda item: item[1])
        logging.info(
            "Peak %s: timestamp=%s, value=%.1f W, slot=%d",
            label.lower(),
            (timestamps[peak_index].isoformat() if timestamps[peak_index] is not None
             else f"local-slot-{peak_index}"),
            peak_value, peak_index,
        )

    log_consumption_peak(labels[0], cons_left_times, cons_left_values)
    log_consumption_peak(labels[1], cons_right_times, cons_right_values)

    logging.info(
        "Wetterdaten via Open-Meteo (lat=%.4f, lon=%.4f): hourly entries=%d",
        api_key.LAT, api_key.LON, len(hourly_map)
    )
    if weather_model:
        logging.info("Weather model: %s", weather_model)
    if sun_today_h is not None or sun_tomorrow_h is not None:
        logging.info(
            "Sonnenstunden: heute=%s h, morgen=%s h",
            _fmt_hours(sun_today_h),
            _fmt_hours(sun_tomorrow_h),
        )
    global SUN_TODAY, SUN_TOMORROW
    SUN_TODAY = sun_today_h
    SUN_TOMORROW = sun_tomorrow_h
    weather_days = aggregate_weather_days(hourly_map)
    for day_name, periods in zip(("today", "tomorrow"), weather_days):
        for (period_name, _start, _end, _offset), period in zip(WEATHER_PERIODS, periods):
            icon_name = get_weather_icon_filename(
                period["code"], period["is_day"]
            ) if period["code"] is not None else "-"
            logging.info(
                "Weather %s %s: temp=%s precipitation=%s code=%s icon=%s",
                day_name, period_name, period["temperature"],
                period["precipitation_probability"], period["code"], icon_name,
            )

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

    # Canvas
    img  = Image.new('1', (w, h), 255)
    d    = ImageDraw.Draw(img)

    # Fonts
    try:
        f_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        f_tiny  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        f_temperature = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19
        )
    except Exception:
        f_bold = f_small = f_tiny = f_temperature = ImageFont.load_default()
    fonts = {'bold': f_bold, 'small': f_small, 'tiny': f_tiny,
             'temperature': f_temperature}

    # Layout
    margin = 10
    top_h = 164
    draw_weather_dashboard(
        d,
        img,
        margin,
        margin,
        w - margin * 2,
        top_h,
        fonts,
        weather_days,
        sun_today_h=sun_today_h,
        sun_tomorrow_h=sun_tomorrow_h,
    )

    # Info-Zeile tiefer und zentriert
    draw_info_box(d, info, fonts, y=top_h + margin + 18, width=w-20)

    # Chart kleiner in der HÃ¶he + Platz fÃ¼r Stunden
    chart_top = top_h + margin + 48
    chart_area = (margin, chart_top, w - margin, h-70)

    draw_two_day_chart(
        img, d, left, right, fonts, labels, chart_area,
        pv_left=pv_left, pv_right=pv_right,
        cons_left=cons_left, cons_right=cons_right,
        cur_dt=info['current_dt'], cur_price=info['current_price']
    )
    draw_live_consumption_box(d, fonts, live_snapshot, w - margin - 142, chart_top - 30)

    footer = dt.datetime.now(LOCAL_TZ).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10, h-10), footer, font=fonts['tiny'], fill=0)

    epd.display(epd.getbuffer(img))
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
