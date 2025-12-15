#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, math, json, requests, datetime as dt, sqlite3, logging
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, numpy as np
from urllib.parse import urlencode
import re

ECO_DEBUG = bool(int(os.getenv("ECO_DEBUG", "0")))
PV_PAT = re.compile(r"(pv|solar|yield|gen|power|input|watt|energy)", re.I)
DUMP_DIR = "/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_dump"

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
        pi = j['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
        logging.info(
            "Tibber Preisinfo via API: heute=%d, morgen=%d, current=%s",
            len(pi.get("today", []) or []),
            len(pi.get("tomorrow", []) or []),
            safe_get(pi, "current", "startsAt", default="-")
        )
        return pi
    except Exception as e:
        raise RuntimeError(f"Tibber Antwort unerwartet: {e}, payload keys: {list(j.keys())}")


def tibber_priceinfo_quarter_range():
    """
    Tibber bietet aktuell keine 15-Minuten-Preise mehr per GraphQL-Enum an.
    Wir verzichten deshalb auf den Versuch und liefern None, sodass die
    Aufrufer automatisch auf die stündlichen Daten zurückfallen.
    """
    logging.info("15-Minuten-Preise werden übersprungen (API-Enum nicht verfügbar)")
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
    return pd.Series(out)

def _pv_series_from_db(slots_dt):
    """Versucht historische PV-Daten aus bekannten Tabellen/Spalten zu lesen."""
    candidates = [
        ("pv_log", "pv_w"),
        ("pv_log", "pv_power"),
        ("pv_data", "pv_w"),
        ("pv_data", "pv_power"),
    ]
    for table, col in candidates:
        s = series_from_db(table, col, slots_dt)
        if s is None:
            continue
        try:
            if len(s) and s.max() > 0:
                return s
        except Exception:
            continue
    return None


def _pv_series_from_ecoflow_history(slots_dt):
    """
    Nutzt die EcoFlow-API (/device/quota/power), um die heutige PV-Kurve
    des Wechselrichters als 15-Minuten-Reihe aufzulösen. Wird nur genutzt,
    wenn keine lokalen DB-Daten verfügbar sind.
    """
    if not slots_dt:
        return None

    target_date = slots_dt[0].date()
    sn_candidates = [
        getattr(api_key, "ECOFLOW_DEVICE_ID", "").strip(),
        getattr(api_key, "ECOFLOW_MIKRO_ID", "").strip(),
    ]
    sn_candidates = [sn for sn in sn_candidates if sn]
    if not sn_candidates:
        return None

    history_series = None
    for sn in sn_candidates:
        try:
            sn_effective = ecoflow_get_main_sn(sn) or sn
        except Exception as e:
            logging.info("EcoFlow main-sn lookup (history) failed: %s", e)
            sn_effective = sn

        try:
            hist = ecoflow_pv_history(sn_effective)
            if hist is None or hist.empty:
                continue
            # Nur verwenden, wenn der Tag zur gewünschten Slots-Date passt
            hist_dates = set(ts.date() for ts in hist.index)
            if target_date not in hist_dates:
                continue

            hist = hist.sort_index()
            values = []
            for ts in slots_dt:
                v = hist.asof(ts) if not hist.empty else None
                values.append(float(0.0 if v is None or pd.isna(v) else v))

            history_series = pd.Series(values)
            logging.info(
                "Nutze EcoFlow PV-Verlaufsdaten für SN %s (%d Punkte)",
                sn_effective,
                len(history_series),
            )
            break
        except Exception as e:
            logging.error("EcoFlow PV history fehlgeschlagen für %s: %s", sn, e)
            continue

    return history_series


def get_pv_series(slots_dt, eco=None):
    """
    Holt historische PV-Daten aus der lokalen DB (falls vorhanden) und fällt
    andernfalls auf die aktuelle EcoFlow-PV-Leistung zurück.
    """
    try:
        from_db = _pv_series_from_db(slots_dt)
        if from_db is not None:
            logging.info("Nutze historische PV-Daten aus der lokalen DB")
            return from_db

        from_history = _pv_series_from_ecoflow_history(slots_dt)
        if from_history is not None:
            return from_history

        if eco is None:
            eco = ecoflow_status_bkw()
        pv_watt = eco.get("pv_input_w_sum") or eco.get("pv_w")
        pv_watt = float(pv_watt or 0.0)
        if ECO_DEBUG:
            logging.info(f"EcoFlow PV aktuell: {pv_watt} W")
        return pd.Series([pv_watt] * len(slots_dt))
    except Exception as e:
        logging.error(f"PV aus EcoFlow fehlgeschlagen: {e}")
        return pd.Series([0.0] * len(slots_dt))


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


def ecoflow_quota_data(sn_main, beginTime_utc, endTime_utc, code):
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/quota/data"
    query = {"sn": sn_main}
    body = {
        "sn": sn_main,
        "params": {
            "beginTime": beginTime_utc,
            "endTime": endTime_utc,
            "code": code,
        },
    }
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, body, content_type="application/json")
    url = f"{base}{path}?{urlencode(query)}"
    r = requests.post(url, headers=hdr, json=body, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    data = j.get("data") if isinstance(j, dict) else None
    count = len(data) if isinstance(data, list) else 0
    unit = None
    try:
        if data and isinstance(data[0], dict):
            unit = data[0].get("unit")
    except Exception:
        pass
    logging.info(
        "EcoFlow quota/data code=%s count=%s unit=%s",
        code,
        count,
        unit,
    )
    if r.status_code == 200 and str(j.get("code")) == "0" and isinstance(data, list):
        return data
    return None

def ecoflow_pv_history(sn):
    """
    Holt die heutige PV-Verlaufsreihe (15-min/5-min) direkt vom EcoFlow BKW.
    Laut BKW-Doku liefert /iot-open/sign/device/quota/power eine Zeitreihe mit
    PV-Leistungswerten. Wir parsen flexibel, um unterschiedliche Feldnamen
    ("pvPower", "power", "value") und Timestamps (Sek./ms) zu unterstützen.
    """
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/device/quota/power"
    query = {"sn": sn}
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, query, content_type=None)
    url = f"{base}{path}?{urlencode(query)}"
    r = requests.get(url, headers=hdr, timeout=12)
    try:
        j = r.json()
    except Exception:
        j = {"raw": r.text}

    if not (r.status_code == 200 and str(j.get("code")) == "0"):
        raise RuntimeError(f"PV history fehlgeschlagen: HTTP {r.status_code}, resp={str(j)[:200]}")

    data = j.get("data") or []
    points = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ts_raw = entry.get("timestamp") or entry.get("time") or entry.get("ts")
        if ts_raw is None:
            continue
        try:
            ts_raw = int(ts_raw)
            if ts_raw > 1e12:  # ms
                ts_dt = dt.datetime.fromtimestamp(ts_raw / 1000.0, tz=dt.timezone.utc)
            else:
                ts_dt = dt.datetime.fromtimestamp(ts_raw, tz=dt.timezone.utc)
            ts_dt = ts_dt.astimezone(LOCAL_TZ)
        except Exception:
            continue

        power = (
            _to_float(entry.get("pvPower"))
            or _to_float(entry.get("power"))
            or _to_float(entry.get("value"))
            or _to_float(entry.get("pv"))
        )
        if power is None:
            continue
        points.append((ts_dt, float(power)))

    if not points:
        raise RuntimeError("PV history leer")

    points.sort(key=lambda p: p[0])
    df = pd.DataFrame(points, columns=["ts", "pv_w"])
    df.set_index("ts", inplace=True)
    # Die API liefert typischerweise 5-Minuten-Slots. Wir glätten auf 15T und füllen.
    df = df.resample("15T").mean().ffill()
    return df["pv_w"]

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
                with open("/home/alex/E-Paper-tibber-Preisanzeige/ecoflow_quota_last.json", "w") as f:
                    json.dump(q_main, f, indent=2)
            except Exception:
                pass
        if sn_micro and ECO_DEBUG and sn_micro != sn_main_effective:
            logging.info("EcoFlow quota/all for micro skipped by default")
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

    pv_src = q_pv or q_main

    # Kerngrößen
    soc     = num(q_main, "cmsBattSoc")
    if soc is not None:
        try: soc = int(round(soc))
        except: pass

    pv_sum  = num(pv_src, "powGetPvSum")
    if pv_sum is None:
        pv_sum = num(pv_src, "powGetPv1InputW", default=None)
        if pv_sum is not None:
            pv_sum += num(pv_src, "powGetPv2InputW", default=0.0)
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
        "load_w": load_w
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
    title_x, title_y = x + 10, y + 5
    d.text((title_x, title_y), title, font=fonts['bold'], fill=0)

    # Batterie
    batt_x, batt_y = x + 10, y + 28
    draw_battery(d, batt_x, batt_y, 90, 28, st.get('soc'), arrow=None, fonts=fonts)

    # Hilfsfunktion
    def fmt_w(v):
        try:
            return f"{int(round(float(v)))} W"
        except Exception:
            return "-"

    text_offset = 10

    # Klarere Zuordnung: Batterie-/Systemleistung, PV-Eingang, Netz, Haushaltslast
    power_w = st.get('power_w') or st.get('gridConnectionPower')
    pv_w    = st.get('pv_input_w_sum') or st.get('powGetPvSum')
    grid_w  = st.get('grid_w') or st.get('powGetSysGrid') or st.get('gridConnectionPower')
    load_w  = st.get('load_w') or st.get('powGetSysLoad')

    # Rechenweg: Leistung + PV-Ertrag + Netz = Last
    base_x = x + w - 140
    op_x   = base_x
    lbl_x  = base_x + 12
    val_x  = base_x + 105
    base_y = y + 6
    row_height = 14

    entries = [
        ("", "Leistung", power_w),          # Batterie (+ Entladen, - Laden)
        ("+", "PV-Ertrag", pv_w),           # Aktuelle PV-Einspeisung
        ("+", "Netz", grid_w)               # Bezug (+) / Einspeisung (-)
    ]

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
        try: return float(np.nanmax(series)) if len(series)>0 else 0
        except: return 0
    pmax = max(vmax_power(pv_left), vmax_power(pv_right),
               vmax_power(cons_left), vmax_power(cons_right))
    sy_pow = H/((pmax or 1)*1.2)

    # Preis-Y-Ticks (nur innerhalb)
    step = 5
    yv = math.floor(vmin/step) * step
    while yv <= vmax:
        yy = _price_to_y(yv)
        if Y0 < yy < Y1:
            d.text((X0-45, yy-7), f"{yv/100:.2f}", font=fonts['tiny'], fill=0)
        yv += step

    def panel(ts_list, val_list, pv_list, cons_list, x0):
        n = len(ts_list)
        if n < 2: return
        xs = [x0 + i*(PW/(n-1)) for i in range(n)]
        # Preis Stufenlinie
        for i in range(n-1):
            x1, y1 = xs[i],   _price_to_y(val_list[i])
            x2, y2 = xs[i+1], _price_to_y(val_list[i+1])
            d.line((x1,y1, x2,y1), fill=0, width=2)
            d.line((x2,y1, x2,y2), fill=0, width=2)
        # PV (historische Linie)
        if pv_list is not None and n == len(pv_list):
            for i in range(n-1):
                y1 = Y1 - pv_list.iloc[i]*sy_pow
                y2 = Y1 - pv_list.iloc[i+1]*sy_pow
                d.line((xs[i], y1, xs[i+1], y2), fill=0, width=1)
        # Verbrauch
        if cons_list is not None and n == len(cons_list):
            for i in range(n-1):
                y1 = Y1 - cons_list.iloc[i]*sy_pow
                y2 = Y1 - cons_list.iloc[i+1]*sy_pow
                draw_dashed_line(d, xs[i], y1, xs[i+1], y2, dash=4, gap=3, width=1)
        # Min/Max Labels
        vmin_i, vmax_i = val_list.index(min(val_list)), val_list.index(max(val_list))
        for idx in (vmin_i, vmax_i):
            xi, yi = xs[idx], _price_to_y(val_list[idx])
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
    d.text((X1-180, Y0-16), "-  Strompreis   ----  Verbrauch", font=fonts['tiny'], fill=0)

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
                py = _price_to_y(cur_price)
                r = 4
                d.ellipse((px - r, py - r, px + r, py + r), fill=0)
                d.text((px + r + 2, py - r - 2), f"{cur_price/100:.2f}", font=fonts['tiny'], fill=0)

# ---------- Main ----------
def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()
    w, h = epd.width, epd.height

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

    info = prepare_info(pi)

    tomorrow = pi.get('tomorrow', [])
    if tomorrow:
        left, right = pi['today'], tomorrow
        labels = ("Heute", "Morgen")
        left_date = dt.date.today()
        right_date = dt.date.today() + dt.timedelta(days=1)
    else:
        left, right = (load_cache(CACHE_YESTERDAY) or {"data": []})['data'], pi['today']
        labels = ("Gestern", "Heute")
        left_date = dt.date.today() - dt.timedelta(days=1)
        right_date = dt.date.today()

    logging.info(
        "Preis-Slots Quelle: %s, linke Achse=%s (%d Werte), rechte Achse=%s (%d Werte)",
        tibber_source,
        labels[0], len(left or []),
        labels[1], len(right or [])
    )

    if quarter_range and quarter_range.get("range", {}).get("nodes"):
        nodes = quarter_range["range"].get("nodes", [])
        left_q = _filter_range_for_date(nodes, left_date)
        right_q = _filter_range_for_date(nodes, right_date)
        if left_q:
            left = left_q
        if right_q:
            right = right_q
    
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

    pv_left  = get_pv_series(tl_dt, eco=eco)
    pv_right = get_pv_series(tr_dt, eco=eco)

    hourly = tibber_hourly_consumption(last=48)
    if hourly:
        logging.info(
            "Tibber Verbrauchsdaten via API: letzte %d Stunden, Start=%s, Ende=%s",
            len(hourly),
            hourly[0][0].strftime("%Y-%m-%d %H:%M"),
            hourly[-1][0].strftime("%Y-%m-%d %H:%M")
        )
    else:
        logging.info("Keine Tibber-Verbrauchsdaten erhalten")
    cons_left  = upsample_hourly_to_quarter(tl_dt, hourly)
    cons_right = upsample_hourly_to_quarter(tr_dt, hourly)

    sun_today, sun_tomorrow = sunshine_hours_both(api_key.LAT, api_key.LON)
    logging.info(
        "Wetterdaten via Open-Meteo (lat=%.4f, lon=%.4f): heute=%.1f h, morgen=%.1f h",
        api_key.LAT, api_key.LON, sun_today, sun_tomorrow
    )

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


if __name__ == "__main__":
    main()
