#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sqlite3, datetime as dt, math, json, argparse, logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Berlin")

DB_FILE = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"
OUT_PNG = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_forecast_tomorrow.png"

# Optional: Open-Meteo sunshine_duration (Sekunden)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def _local_day_bounds_utc(date_local: dt.date):
    begin_local = dt.datetime.combine(date_local, dt.time.min, tzinfo=LOCAL_TZ)
    end_local = dt.datetime.combine(date_local, dt.time.max, tzinfo=LOCAL_TZ)
    begin_utc = int(begin_local.astimezone(dt.timezone.utc).timestamp())
    end_utc = int(end_local.astimezone(dt.timezone.utc).timestamp())
    return begin_local, end_local, begin_utc, end_utc


def _read_pv_day_df(date_local: dt.date, db_file=DB_FILE) -> pd.DataFrame:
    _, _, begin_utc, end_utc = _local_day_bounds_utc(date_local)
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(
            "SELECT ts, pv_sum_w FROM pv_log WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
            conn,
            params=(begin_utc, end_utc),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    df["pv_sum_w"] = pd.to_numeric(df["pv_sum_w"], errors="coerce").fillna(0.0)
    df["pv_sum_w"] = df["pv_sum_w"].clip(lower=0.0)
    return df


def pv_daily_energy_wh_from_db(date_local: dt.date, db_file=DB_FILE):
    df = _read_pv_day_df(date_local, db_file=db_file)
    if df.empty:
        return None
    s = df["pv_sum_w"].resample("15T").mean().ffill().fillna(0.0).clip(lower=0.0)
    energy_wh = float((s * 0.25).sum())  # 15 min = 0.25h
    return energy_wh


def pv_profile_normalized_from_db(date_local: dt.date, slots_dt_template: pd.DatetimeIndex, db_file=DB_FILE):
    df = _read_pv_day_df(date_local, db_file=db_file)
    if df.empty:
        return None

    s = df["pv_sum_w"].resample("15T").mean().ffill().fillna(0.0).clip(lower=0.0)

    # auf template mappen (asof)
    out_vals = []
    for t in slots_dt_template:
        v = s.asof(t)
        out_vals.append(float(0.0 if v is None or pd.isna(v) else v))
    prof_w = pd.Series(out_vals, index=slots_dt_template).clip(lower=0.0)

    energy_wh = float((prof_w * 0.25).sum())
    if energy_wh < 50.0:  # zu wenig PV -> Tag nicht verwenden
        return None

    denom = float(prof_w.sum())
    if denom <= 0:
        return None

    prof_norm = prof_w / denom  # Summe=1 über den Tag (Form)
    return prof_norm


def _make_slots_for_date(date_local: dt.date) -> pd.DatetimeIndex:
    begin = dt.datetime.combine(date_local, dt.time.min, tzinfo=LOCAL_TZ)
    # 96 Slots à 15min
    return pd.date_range(begin, periods=96, freq="15T", tz=LOCAL_TZ)


def sunshine_hours_both(lat: float, lon: float, model: str = "icon_seamless"):
    """
    Liefert sunshine_duration in Stunden für heute und morgen.
    """
    import requests
    url = (
        f"{OPEN_METEO_URL}"
        f"?latitude={lat}&longitude={lon}"
        "&daily=sunshine_duration&timezone=Europe%2FBerlin"
        f"&models={model}"
    )
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    j = r.json()
    arr = (j.get("daily", {}).get("sunshine_duration")) or []
    def _h(idx):
        try:
            sec = arr[idx]
            return 0.0 if sec is None else float(sec) / 3600.0
        except Exception:
            return 0.0
    return _h(0), _h(1), j


def pv_forecast_series_for_tomorrow(
    lat: float = None,
    lon: float = None,
    db_file=DB_FILE,
    lookback_profiles_days: int = 14,
    lookback_energy_days: int = 7,
    force_scale: float = None,
):
    today = dt.datetime.now(tz=LOCAL_TZ).date()
    tomorrow = today + dt.timedelta(days=1)
    slots_tom = _make_slots_for_date(tomorrow)

    # 1) Profile sammeln (letzte 14 Tage)
    profiles = []
    used_days = []
    for k in range(1, lookback_profiles_days + 1):
        day = tomorrow - dt.timedelta(days=k)
        slots_day = _make_slots_for_date(day)  # gleicher Raster, anderes Datum
        p = pv_profile_normalized_from_db(day, slots_day, db_file=db_file)
        if p is not None and len(p) == 96:
            # auf morgen-index umhängen (nur die Form zählt)
            p2 = pd.Series(p.to_numpy(), index=slots_tom)
            profiles.append(p2)
            used_days.append(day)

    if not profiles:
        logging.warning("Keine brauchbaren Profile in DB gefunden (letzte %d Tage).", lookback_profiles_days)
        empty_series = pd.Series([np.nan] * len(slots_tom), index=slots_tom)
        return empty_series, {
            "tomorrow": str(tomorrow),
            "base_wh": 0.0,
            "scale": 1.0,
            "forecast_wh": 0.0,
            "used_days": [],
            "sun_today_h": None,
            "sun_tomorrow_h": None,
            "profile_sum": 0.0,
            "energy_check_wh": 0.0,
            "peak_w": 0.0,
            "peak_time": "n/a",
            "weather": None,
        }

    prof_df = pd.concat(profiles, axis=1)  # columns=days
    median_profile = prof_df.median(axis=1).clip(lower=0.0)

    # normieren auf Summe=1 (Sicherheitsnetz)
    ssum = float(median_profile.sum())
    if ssum <= 0:
        empty_series = pd.Series([np.nan] * len(slots_tom), index=slots_tom)
        return empty_series, {
            "tomorrow": str(tomorrow),
            "base_wh": 0.0,
            "scale": 1.0,
            "forecast_wh": 0.0,
            "used_days": [d.isoformat() for d in used_days],
            "sun_today_h": None,
            "sun_tomorrow_h": None,
            "profile_sum": 0.0,
            "energy_check_wh": 0.0,
            "peak_w": 0.0,
            "peak_time": "n/a",
            "weather": None,
        }
    median_profile = median_profile / ssum
    profile_sum = float(median_profile.sum())

    # 2) Basisenergie (Wh) aus letzten 7 Tagen
    energies = []
    for k in range(1, lookback_energy_days + 1):
        day = today - dt.timedelta(days=k)
        e = pv_daily_energy_wh_from_db(day, db_file=db_file)
        if e is not None and e >= 50.0:
            energies.append(float(e))
    base_wh = float(np.mean(energies)) if energies else 0.0

    # 3) Wetter-Skalierung
    scale = 1.0
    weather = None
    sun_today_h = None
    sun_tomorrow_h = None
    if force_scale is not None:
        scale = float(force_scale)
    elif lat is not None and lon is not None:
        try:
            sun_today, sun_tom, weather = sunshine_hours_both(lat, lon)
            sun_today_h = float(sun_today)
            sun_tomorrow_h = float(sun_tom)
            # robust: bei sehr wenig Sonne heute clamp denominator
            denom = max(float(sun_today), 0.2)
            scale = float(sun_tom) / denom
            scale = max(0.2, min(1.8, scale))
        except Exception as e:
            logging.info("Wetter-Skalierung nicht verfügbar (%s) -> scale=1.0", e)
            scale = 1.0

    forecast_wh = base_wh * scale

    # 4) In W-Serie (15-min) zurück
    # Ziel: Sum(W*0.25)=forecast_wh => W = profile * (forecast_wh/0.25)
    w_series = median_profile * (forecast_wh / 0.25)
    energy_check_wh = float((w_series.clip(lower=0.0) * 0.25).sum())
    peak_series = w_series.fillna(0.0).clip(lower=0.0)
    if len(peak_series) > 0:
        peak_w = float(peak_series.max())
        peak_time = peak_series.idxmax().strftime("%H:%M")
    else:
        peak_w = 0.0
        peak_time = "n/a"

    meta = {
        "tomorrow": str(tomorrow),
        "base_wh": base_wh,
        "scale": scale,
        "forecast_wh": forecast_wh,
        "used_days": [d.isoformat() for d in used_days],
        "sun_today_h": sun_today_h,
        "sun_tomorrow_h": sun_tomorrow_h,
        "profile_sum": profile_sum,
        "energy_check_wh": energy_check_wh,
        "peak_w": peak_w,
        "peak_time": peak_time,
        "weather": weather,
    }
    return w_series, meta


def render_forecast_png(series_w: pd.Series, meta: dict, outfile: str, width=800, height=480, y_max=600.0):
    """
    Simple Vorschau-Grafik: Achse 0..y_max W, Linie für morgen.
    """
    img = Image.new("1", (width, height), 255)
    d = ImageDraw.Draw(img)

    # Fonts
    try:
        f_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        f_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        f_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        f_m = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except Exception:
        f_b = f_s = f_t = f_m = ImageFont.load_default()

    # Layout
    margin = 40
    X0 = margin
    X1 = width - margin

    # Title
    title = f"PV-Prognose (morgen) – {meta.get('tomorrow','')}"
    d.text((margin, 18), title, font=f_b, fill=0)

    # Forecast steps block
    base_wh = float(meta.get("base_wh", 0.0))
    scale = float(meta.get("scale", 1.0))
    fc_wh = float(meta.get("forecast_wh", 0.0))
    used_days = meta.get("used_days", [])
    sun_today_h = meta.get("sun_today_h", None)
    sun_tomorrow_h = meta.get("sun_tomorrow_h", None)
    profile_sum = float(meta.get("profile_sum", 0.0))
    peak_w = float(meta.get("peak_w", 0.0))
    energy_check_wh = float(meta.get("energy_check_wh", 0.0))

    used_days_text = ""
    if used_days:
        used_days_text = f" ({', '.join(used_days[-5:])})"

    def _fmt_h(val):
        return "n/a" if val is None else f"{val:.1f}"

    steps_lines = [
        "Forecast Steps",
        f"DB days used (profiles): {len(used_days)}{used_days_text}",
        f"base_wh (mean last 7 valid days): {base_wh:.0f} Wh",
        f"sun_today: {_fmt_h(sun_today_h)} h   sun_tomorrow: {_fmt_h(sun_tomorrow_h)} h",
        f"scale = clamp(sun_tomorrow / max(sun_today,0.2), 0.2..1.8): {scale:.2f}",
        f"forecast_wh = base_wh * scale: {fc_wh:.0f} Wh",
        f"profile normalization: sum(profile)={profile_sum:.3f}",
        f"peak_w (max of forecast series): {peak_w:.0f} W",
        f"y_max: {y_max:.0f} W",
    ]
    block_x = margin
    block_y = 40
    line_h = 14
    for i, line in enumerate(steps_lines):
        d.text((block_x, block_y + i * line_h), line, font=f_m, fill=0)

    Y0 = block_y + len(steps_lines) * line_h + 10
    Y1 = height - 60
    W, H = X1 - X0, Y1 - Y0

    # Axes box
    d.rectangle((X0, Y0, X1, Y1), outline=0, width=2)

    # y ticks
    y_max = float(max(1.0, y_max))
    for yv in [0, 100, 200, 300, 400, 500, 600]:
        if yv > y_max:
            continue
        yy = Y1 - (yv / y_max) * H
        d.line((X0 - 5, yy, X0, yy), fill=0, width=1)
        d.text((X0 - 35, yy - 8), str(yv), font=f_t, fill=0)

    # x ticks (hours)
    ts = list(series_w.index)
    n = len(ts)
    if n >= 2:
        for i, t in enumerate(ts):
            if t.minute == 0:
                x = X0 + (i / (n - 1)) * W
                d.line((x, Y1, x, Y1 + 4), fill=0, width=1)
                d.text((x - 10, Y1 + 8), t.strftime("%H"), font=f_t, fill=0)

    # Line
    arr = series_w.to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0.0, y_max)

    pts = []
    for i, v in enumerate(arr):
        x = X0 + (i / (n - 1)) * W
        y = Y1 - (v / y_max) * H
        pts.append((x, y))

    # draw polyline
    last = None
    for p in pts:
        if last is not None:
            d.line((last[0], last[1], p[0], p[1]), fill=0, width=2)
        last = p

    # Peak marker
    if len(arr) > 0:
        peak_idx = int(np.argmax(arr))
        peak_x = X0 + (peak_idx / (n - 1)) * W
        peak_y = Y1 - (arr[peak_idx] / y_max) * H
        r = 3
        d.ellipse((peak_x - r, peak_y - r, peak_x + r, peak_y + r), outline=0, fill=0)
        label = f"peak {peak_w:.0f}W @ {meta.get('peak_time','n/a')}"
        d.text((peak_x + 6, max(Y0, peak_y - 12)), label, font=f_t, fill=0)

    # Energy check
    d.text((margin, height - 46), f"Energy check: sum(W*0.25h)={energy_check_wh:.0f} Wh", font=f_t, fill=0)

    # Footer
    now = dt.datetime.now(tz=LOCAL_TZ).strftime("%H:%M %d.%m.%Y")
    d.text((margin, height - 30), f"generated {now}", font=f_t, fill=0)

    img.save(outfile)
    return outfile


def show_on_epd(png_path: str):
    # Waveshare EPD 7.5 V2 (wie in deinem Display-Code)
    import sys
    sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
    sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
    from waveshare_epd import epd7in5_V2

    epd = epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img = Image.open(png_path).convert("1")

    # Auf Displaygröße bringen
    w, h = epd.width, epd.height
    if img.size != (w, h):
        img = img.resize((w, h))

    epd.display(epd.getbuffer(img))
    epd.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_FILE)
    ap.add_argument("--out", default=OUT_PNG)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=600.0)
    ap.add_argument("--epd", action="store_true", help="Bild aufs Waveshare EPD 7.5 V2 ausgeben")
    ap.add_argument("--force-scale", type=float, default=None, help="Optional: Wetter-Scale erzwingen (Debug)")
    args = ap.parse_args()

    series, meta = pv_forecast_series_for_tomorrow(
        lat=args.lat, lon=args.lon, db_file=args.db, force_scale=args.force_scale
    )

    # Meta zusätzlich als JSON ablegen (Debug)
    meta_path = os.path.splitext(args.out)[0] + ".json"
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logging.info("Meta geschrieben: %s", meta_path)
    except Exception as e:
        logging.info("Meta write failed: %s", e)

    out = render_forecast_png(series, meta, args.out, width=800, height=480, y_max=args.ymax)
    logging.info("PNG geschrieben: %s", out)

    if args.epd:
        show_on_epd(out)
        logging.info("Auf EPD angezeigt.")

if __name__ == "__main__":
    main()
