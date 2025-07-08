#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, json, datetime
import pandas as pd
import matplotlib.pyplot as plt

# 1) Wo liegen deine Logs? (Anpassen, falls nötig)
DATA_DIR = "/home/alex/realdata_logs"

# 2) Datum definieren
today     = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def load_pv_day(date):
    """Lädt alle JSON-Logs eines Tages, fasst PV-Leistung auf 15-Min.-Intervalle zusammen."""
    frames = []
    for fn in os.listdir(DATA_DIR):
        if not fn.startswith(f"realdata-{date.isoformat()}"): continue
        with open(os.path.join(DATA_DIR, fn)) as f:
            data = json.load(f)
        # Zeitstempel
        ts = pd.to_datetime(data["timestamp"], unit="s")
        # Summe über alle PV-Ports
        pv_power = sum(item.get("power", 0) for item in data.get("pv_data", []))
        frames.append(pd.DataFrame({"pv_power": pv_power}, index=[ts]))
    if not frames:
        return pd.DataFrame(columns=["pv_power"])
    df = pd.concat(frames).sort_index()
    # Resample auf 15-Minuten-Intervalle (here: Summe, du kannst .mean() nehmen)
    return df.resample("15T").sum()

# 3) DataFrames für gestern und heute
df_y = load_pv_day(yesterday)
df_t = load_pv_day(today)

# 4) Chart zeichnen
plt.figure(figsize=(10,4))
plt.plot(df_y.index, df_y["pv_power"], label=yesterday.isoformat())
plt.plot(df_t.index, df_t["pv_power"], label=today     .isoformat())
plt.xlabel("Uhrzeit")
plt.ylabel("PV-Leistung (W)")
plt.title("PV-Erzeugung: gestern vs. heute (15-Minuten-Intervalle)")
plt.legend()
plt.tight_layout()

# 5) Speichern
out_png = f"/home/alex/epaper/pv_chart_{today.isoformat()}.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png)
print(f"Chart gespeichert in {out_png}")
