#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime
import random
from PIL import Image, ImageDraw, ImageFont

# ZoneInfo für Zeitzonenkonversion (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Für ältere Python-Versionen

# Lokale Zeitzone (anpassen, falls nötig)
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare-Treiber anpassen (falls nötig)
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')

from waveshare_epd import epd7in5_V2
import api_key

# --- Caching-Konstanten ---
CACHE_FILE_TODAY = 'cached_today_price.json'
CACHE_FILE_YESTERDAY = 'cached_yesterday_price.json'

#####################################
# Hilfsfunktionen für Caching und API
#####################################
def save_cache(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def update_price_cache(price_data):
    today_date_str = datetime.date.today().isoformat()
    cached_today = load_cache(CACHE_FILE_TODAY)
    if cached_today is None or cached_today.get('date') != today_date_str:
        if cached_today is not None:
            save_cache(cached_today, CACHE_FILE_YESTERDAY)
        cache_data = {"date": today_date_str, "data": price_data['today']}
        save_cache(cache_data, CACHE_FILE_TODAY)

def get_cached_yesterday_price():
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    headers = {
        "Authorization": "Bearer " + api_key.API_KEY,
        "Content-Type": "application/json"
    }
    query = """
    {
      viewer {
        homes {
          currentSubscription {
            priceInfo {
              today { total startsAt }
              tomorrow { total startsAt }
              current { total startsAt }
            }
          }
        }
      }
    }
    """
    data = {"query": query}
    response = requests.post("https://api.tibber.com/v1-beta/gql", json=data, headers=headers)
    return response.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def get_consumption_data():
    headers = {
        "Authorization": "Bearer " + api_key.API_KEY,
        "Content-Type": "application/json"
    }
    query = """
    {
      viewer {
        homes {
          consumption(resolution: HOURLY, last: 48) {
            nodes {
              from
              to
              consumption
              cost
            }
          }
        }
      }
    }
    """
    data = {"query": query}
    response = requests.post("https://api.tibber.com/v1-beta/gql", json=data, headers=headers)
    try:
        response_json = response.json()
    except Exception as e:
        print("Fehler beim Parsen der API-Antwort:", e)
        return None
    if "data" not in response_json:
        print("Fehler: 'data'-Schlüssel fehlt:", response_json)
        return None
    if "errors" in response_json:
        print("API-Fehler:", response_json["errors"])
        return None
    try:
        consumption_data = response_json['data']['viewer']['homes'][0]['consumption']['nodes']
    except (KeyError, IndexError) as e:
        print("Fehler beim Zugriff auf Verbrauchsdaten:", e)
        return None
    return consumption_data

def extract_yesterday_consumption(consumption_data):
    yesterday_date = datetime.date.today() - datetime.timedelta(days=1)
    total_consumption = 0.0
    total_cost = 0.0
    for record in consumption_data:
        record_from = datetime.datetime.fromisoformat(record['from']).astimezone(local_tz).date()
        if record_from == yesterday_date:
            total_consumption += record.get('consumption', 0.0)
            total_cost += record.get('cost', 0.0)
    return total_consumption, total_cost

def filter_yesterday_consumption(consumption_data):
    yesterday_date = datetime.date.today() - datetime.timedelta(days=1)
    filtered = []
    if consumption_data is None:
        return filtered
    for record in consumption_data:
        try:
            dt = datetime.datetime.fromisoformat(record['from']).astimezone(local_tz).date()
            if dt == yesterday_date:
                filtered.append(record)
        except Exception:
            pass
    return filtered

def prepare_data(price_data):
    timestamps = []
    labels = []
    prices_cents = []
    day_boundary_index = len(price_data['today'])
    for day in ['today', 'tomorrow']:
        for slot in price_data[day]:
            dt_obj = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            timestamps.append(dt_obj)
            labels.append(dt_obj.strftime("%d.%m %Hh"))
            prices_cents.append(slot['total'] * 100.0)
    current_price = price_data['current']['total'] * 100.0
    current_time_obj = datetime.datetime.fromisoformat(price_data['current']['startsAt']).astimezone(local_tz)
    current_index = min(range(len(timestamps)), key=lambda i: abs((timestamps[i]-current_time_obj).total_seconds()))
    today_cents = [slot['total'] * 100.0 for slot in price_data['today']]
    lowest_today = min(today_cents)
    highest_today = max(today_cents)
    future_slots = [(timestamps[i], prices_cents[i])
                    for i in range(len(prices_cents)) if timestamps[i] >= current_time_obj]
    if future_slots:
        lowest_future_time, lowest_future_val = min(future_slots, key=lambda x: x[1])
        hours_to_lowest = round((lowest_future_time - current_time_obj).total_seconds()/3600)
    else:
        lowest_future_val = 0
        hours_to_lowest = 0
    data_min = min(prices_cents) - 1 if prices_cents else 0
    data_max = max(prices_cents) + 1 if prices_cents else 0
    return {"timestamps": timestamps, "labels": labels, "prices_cents": prices_cents,
            "day_boundary_index": day_boundary_index, "current_price": current_price,
            "current_index": current_index, "lowest_today": lowest_today, "highest_today": highest_today,
            "lowest_future_val": lowest_future_val, "hours_to_lowest": hours_to_lowest,
            "data_min": data_min, "data_max": data_max}

##########################################
# Zeichnungsfunktionen (Charts, Marker, etc.)
##########################################
def draw_dashed_line(draw, x1, y1, x2, y2, fill=0, width=1, dash_length=4, gap_length=4):
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    dash_count = int(distance/(dash_length+gap_length)) if distance else 0
    for d in range(dash_count+1):
        start = d*(dash_length+gap_length)
        end = start+dash_length
        if end > distance:
            end = distance
        ratio_start = start/distance if distance else 0
        ratio_end = end/distance if distance else 0
        sx1 = x1 + dx*ratio_start
        sy1 = y1 + dy*ratio_start
        sx2 = x1 + dx*ratio_end
        sy2 = y1 + dy*ratio_end
        draw.line((sx1, sy1, sx2, sy2), fill=fill, width=width)

def get_stepped_marker_position(now_local, times_list, x_positions, values_list, chart_y_bottom, val_min, scale_y):
    n = len(times_list)
    if n == 0:
        return (0, chart_y_bottom, -1)
    if now_local < times_list[0]:
        return (x_positions[0], chart_y_bottom - (values_list[0]-val_min)*scale_y, 0)
    if now_local >= times_list[-1]:
        return (x_positions[-1], chart_y_bottom - (values_list[-1]-val_min)*scale_y, n-1)
    for i in range(n-1):
        if times_list[i] <= now_local < times_list[i+1]:
            total_secs = (times_list[i+1]-times_list[i]).total_seconds()
            elapsed_secs = (now_local-times_list[i]).total_seconds()
            frac = elapsed_secs/total_secs
            x = x_positions[i] + frac*(x_positions[i+1]-x_positions[i])
            # Vertikal nehmen wir den diskreten (abgelaufenen) Stundenwert
            y = chart_y_bottom - (values_list[i]-val_min)*scale_y
            return (x, y, i)
    return (x_positions[-1], chart_y_bottom - (values_list[-1]-val_min)*scale_y, n-1)

def draw_two_day_chart(draw, left_data, left_type,_
