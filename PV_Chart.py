#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, json, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = "/home/alex/realdata_logs"
today     = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def load_pv_day(date):
    frames = []
    for fn in os.listdir(DATA_DIR):
        if not fn.startswith(f"realdata-{date.isoformat()}"): continue
        with open(os.path.join(DATA_DIR, fn)) as f:
            data = json.load(f)
        ts = pd.to_datetime(data["timestamp"], unit="s")
        pv_power = sum(item.get("power", 0) for item in data.get("pv_data", []))
        frames.append(pd.DataFrame({"pv_power": pv_power}, index=[ts]))
    if not frames:
        return pd.DataFrame(columns=["pv_power"])
    df = pd.concat(frames).sort_index()
    return df.resample("15T").sum()

# Daten laden
df_y = load_pv_day(yesterday)
df_t = load_pv_day(today)

# Figure mit 2 Panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Panel 1: gestern
ax1.plot(df_y.index, df_y["pv_power"], marker='o')
ax1.set_title(f"PV-Erzeugung {yesterday.isoformat()}")
ax1.set_xlabel("Uhrzeit")

# Panel 2: heute
ax2.plot(df_t.index, df_t["pv_power"], marker='o', color='orange')
ax2.set_title(f"PV-Erzeugung {today.isoformat()}")
ax2.set_xlabel("Uhrzeit")

# Gemeinsame Y-Achse
ax1.set_ylabel("Leistung (W)")

# Zeit-Achsen formatieren: Major-Stunden, Minor-15-Min
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(which='minor', axis='x', linestyle=':', alpha=0.5)

plt.tight_layout()
out_png = f"/home/alex/epaper/pv_chart_{today.isoformat()}.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png)
print("Chart gespeichert in", out_png)
