#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# — Konfiguration —
HOST     = "192.168.178.119"                     # IP deines Hoymiles-Gateways
DATA_DIR = "/home/alex/realdata_logs"            # wo deine JSON-Logs liegen
OUT_DIR  = "/home/alex/epaper"                   # wo das PNG gespeichert wird

today     = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def fetch_current_and_log():
    """
    Ruft die aktuellen PV-Daten via CLI ab und speichert
    sie als JSON-Logfile in DATA_DIR.
    """
    cmd = [
        "hoymiles-wifi",
        "--host", HOST,
        "--disable-interactive",
        "--as-json",
        "get-real-data-new"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    proc.check_returncode()  # Exception, falls der CLI-Aufruf fehlschlägt

    data = json.loads(proc.stdout)
    # Timestamp in pandas-kompatibles Format
    ts = pd.to_datetime(data["timestamp"], unit="s")
    # Datei-Pfad: realdata-YYYY-MM-DD-HHMM.json
    fn = os.path.join(
        DATA_DIR,
        f"realdata-{ts.date().isoformat()}-{ts.strftime('%H%M')}.json"
    )
    # Ordner anlegen, falls nicht vorhanden
    os.makedirs(DATA_DIR, exist_ok=True)
    # Logfile speichern
    with open(fn, "w") as f:
        json.dump(data, f)
    return

def load_pv_day(date):
    """
    Lädt alle JSON-Logs eines Tages aus DATA_DIR,
    summiert die PV-Leistung aus allen Ports und
    resampled auf 15-Minuten-Intervalle.
    """
    frames = []
    prefix = f"realdata-{date.isoformat()}"
    for fn in os.listdir(DATA_DIR):
        if not fn.startswith(prefix):
            continue
        path = os.path.join(DATA_DIR, fn)
        with open(path) as f:
            data = json.load(f)
        ts = pd.to_datetime(data["timestamp"], unit="s")
        pv_power = sum(item.get("power", 0) for item in data.get("pv_data", []))
        frames.append(pd.DataFrame({"pv_power": pv_power}, index=[ts]))
    if not frames:
        return pd.DataFrame(columns=["pv_power"])
    df = pd.concat(frames).sort_index()
    return df.resample("15T").sum()

def main():
    # 1) Aktuellen PV-Wert holen und loggen
    try:
        fetch_current_and_log()
    except Exception as e:
        print("WARNUNG: Live-Abruf fehlgeschlagen:", e)

    # 2) Daten laden
    df_y = load_pv_day(yesterday)
    df_t = load_pv_day(today)

    # 3) Ausgabeordner und Dateiname
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, f"pv_chart_{today.isoformat()}.png")

    # 4) Figure mit 2 Panels erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax1.plot(df_y.index, df_y["pv_power"], marker='o')
    ax1.set_title(f"PV-Erzeugung {yesterday.isoformat()}")
    ax1.set_xlabel("Uhrzeit")

    ax2.plot(df_t.index, df_t["pv_power"], marker='o', color='orange')
    ax2.set_title(f"PV-Erzeugung {today.isoformat()}")
    ax2.set_xlabel("Uhrzeit")

    ax1.set_ylabel("Leistung (W)")

    # 5) X-Achsen-Begrenzung: 00:00–23:59 gestern / 00:00–jetzt heute
    start_y = datetime.datetime.combine(yesterday, datetime.time(0, 0))
    end_y   = datetime.datetime.combine(yesterday, datetime.time(23, 59))
    start_t = datetime.datetime.combine(today,     datetime.time(0, 0))
    end_t   = datetime.datetime.today()

    ax1.set_xlim(start_y, end_y)
    ax2.set_xlim(start_t, end_t)

    # 6) Ticks & Formatierung (Major = 2h, Minor = 15min)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(which='minor', axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Chart gespeichert in {out_png}")

if __name__ == "__main__":
    main()
