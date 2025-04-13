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

def save_cache(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def update_price_cache(price_data):
    """
    Speichert die heutigen Preisdaten (price_data['today']) in einem Cache.
    Falls im Cache bereits Daten eines anderen Tages vorliegen, werden diese
    als gestrige Daten abgelegt.
    """
    today_date_str = datetime.date.today().isoformat()
    cached_today = load_cache(CACHE_FILE_TODAY)
    if cached_today is None or cached_today.get('date') != today_date_str:
        if cached_today is not None:
            save_cache(cached_today, CACHE_FILE_YESTERDAY)
        cache_data = {"date": today_date_str, "data": price_data['today']}
        save_cache(cache_data, CACHE_FILE_TODAY)

def get_cached_yesterday_price():
    """
    Lädt die gestrigen Preisdaten aus dem Cache (falls vorhanden).
    """
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    """
    Fragt die Tibber-API ab und gibt das price_data-Dictionary zurück.
    Es werden "today" und "tomorrow" (sowie "current") abgefragt.
    """
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
    """
    Fragt die Tibber-API für Verbrauchsdaten ab (stündlich, letzte 48 Stunden).
    Die Query wird angepasst, um die Daten aus dem 'nodes'-Feld zu holen.
    """
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
        print("Fehler in der API-Antwort, 'data'-Schlüssel fehlt:", response_json)
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
    """
    Berechnet aus den Verbrauchsdaten (Liste von stündlichen Records) den
    Gesamtverbrauch und die Gesamtkosten für gestern.
    """
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
    """
    Filtert aus den Verbrauchsdaten jene Records, die zu gestern gehören.
    """
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
    """
    Bereitet die API-Daten auf:
      - Listen mit Timestamps, Labels, Preisen in Cent
      - Index des aktuellen Preises
      - Tageshoch/-tief (nur 'today')
      - Globales Min/Max (für Achsen-Skalierung)
      - Stunden bis zum nächsten Tiefstpreis
    Gibt ein Dictionary mit allen Werten zurück.
    (Wird für die Info-Box unten genutzt.)
    """
    timestamps = []
    labels = []
    prices_cents = []
    day_boundary_index = len(price_data['today'])
    for day in ['today', 'tomorrow']:
        for slot in price_data[day]:
            # Konvertiere in lokale Zeit
            dt_obj = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            timestamps.append(dt_obj)
            labels.append(dt_obj.strftime("%d.%m %Hh"))
            prices_cents.append(slot['total'] * 100.0)
    current_price = price_data['current']['total'] * 100.0
    current_time_obj = datetime.datetime.fromisoformat(price_data['current']['startsAt']).astimezone(local_tz)
    current_index = min(
        range(len(timestamps)),
        key=lambda i: abs((timestamps[i] - current_time_obj).total_seconds())
    )
    today_cents = [slot['total'] * 100.0 for slot in price_data['today']]
    lowest_today = min(today_cents)
    highest_today = max(today_cents)
    future_slots = [
        (timestamps[i], prices_cents[i])
        for i in range(len(prices_cents))
        if timestamps[i] >= current_time_obj
    ]
    if future_slots:
        lowest_future_time, lowest_future_val = min(future_slots, key=lambda x: x[1])
        hours_to_lowest = round((lowest_future_time - current_time_obj).total_seconds() / 3600)
    else:
        lowest_future_val = 0
        hours_to_lowest = 0
    data_min = min(prices_cents) - 1 if prices_cents else 0
    data_max = max(prices_cents) + 1 if prices_cents else 0
    return {
        "timestamps": timestamps,
        "labels": labels,
        "prices_cents": prices_cents,
        "day_boundary_index": day_boundary_index,
        "current_price": current_price,
        "current_index": current_index,
        "lowest_today": lowest_today,
        "highest_today": highest_today,
        "lowest_future_val": lowest_future_val,
        "hours_to_lowest": hours_to_lowest,
        "data_min": data_min,
        "data_max": data_max
    }

def draw_dashed_line(draw, x1, y1, x2, y2, fill=0, width=1, dash_length=4, gap_length=4):
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    dash_count = int(distance / (dash_length + gap_length)) if distance else 0
    for d in range(dash_count + 1):
        start = d * (dash_length + gap_length)
        end = start + dash_length
        if end > distance:
            end = distance
        ratio_start = start / distance if distance else 0
        ratio_end = end / distance if distance else 0
        sx1 = x1 + dx * ratio_start
        sy1 = y1 + dy * ratio_start
        sx2 = x1 + dx * ratio_end
        sy2 = y1 + dy * ratio_end
        draw.line((sx1, sy1, sx2, sy2), fill=fill, width=width)

def get_stepped_marker_position(now_local, times_list, x_positions, values_list,
                                chart_y_bottom, val_min, scale_y):
    """
    Sucht das passende Stundenintervall [i, i+1) für now_local in times_list und
    berechnet:
      - x-Position: linear zwischen x_positions[i] und x_positions[i+1] 
        (→ minütliche Wanderung)
      - y-Position: diskreter Preis von values_list[i] (Stufe)
    Falls now_local < times_list[0], wird i=0 genommen (links).
    Falls now_local >= times_list[-1], wird i=len(times_list)-1 genommen (rechts).
    Gibt (x_marker, y_marker, price_stepped_index) zurück.
    """
    n = len(times_list)
    if n == 0:
        # Keine Daten -> Kein Marker
        return (0, chart_y_bottom, -1)
    
    # Falls Zeit < erstes Interval, an den linken Rand
    if now_local < times_list[0]:
        x = x_positions[0]
        y = chart_y_bottom - (values_list[0] - val_min) * scale_y
        return (x, y, 0)
    
    # Falls Zeit > letztes Interval, an den rechten Rand
    if now_local >= times_list[-1]:
        i = n - 1
        x = x_positions[i]
        y = chart_y_bottom - (values_list[i] - val_min) * scale_y
        return (x, y, i)
    
    # Interval [i, i+1) suchen
    for i in range(n - 1):
        if times_list[i] <= now_local < times_list[i+1]:
            total_secs = (times_list[i+1] - times_list[i]).total_seconds()
            elapsed_secs = (now_local - times_list[i]).total_seconds()
            frac = elapsed_secs / total_secs  # 0..1
            # X interpoliert
            x = x_positions[i] + frac * (x_positions[i+1] - x_positions[i])
            # Y = diskreter Preis der abgelaufenen Stunde i
            y = chart_y_bottom - (values_list[i] - val_min) * scale_y
            return (x, y, i)
    
    # Falls wir hier rausfallen (edge case), nimm letztes Intervall
    i = n - 1
    x = x_positions[i]
    y = chart_y_bottom - (values_list[i] - val_min) * scale_y
    return (x, y, i)

def draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode):
    """
    Zeichnet einen 2-Panel-Chart.
    Im Future-Modus:
      - Linkes Panel zeigt den Preis heute inkl. Datenbeschriftung (Tiefst- und Hochtpreis)
        und markiert den aktuellen Preis.
      - Rechtes Panel zeigt den Preis morgen (bei Future) bzw. den Preis heute im Historical-Modus.
    """
    import datetime
    chart_x_start = 60
    chart_x_end = 800
    chart_y_top = 50
    chart_y_bottom = 400
    chart_width = chart_x_end - chart_x_start
    chart_height = chart_y_bottom - chart_y_top
    panel_width = chart_width / 2

    # --- Linkes Panel (Preis heute oder gestriger Verbrauch + Preis) ---
    times_left = []
    values_left = []
    if left_type == "combo":
        # Bei "combo" erwarten wir, dass left_data ein Dictionary mit Schlüsseln "price" und "consumption" ist.
        for slot in left_data["price"]:
            dt_obj = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            times_left.append(dt_obj)
            values_left.append(slot['total'] * 100.0)
    elif left_type == "price":
        for slot in left_data:
            dt_obj = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            times_left.append(dt_obj)
            values_left.append(slot['total'] * 100.0)
    else:
        # left_type == "consumption"
        for slot in left_data:
            dt_obj = datetime.datetime.fromisoformat(slot['from']).astimezone(local_tz)
            times_left.append(dt_obj)
            values_left.append(slot['consumption'])
    n_left = len(values_left)

    # --- Rechtes Panel (Preis morgen bzw. Preis heute im Historical-Modus) ---
    times_right = []
    values_right = []
    for slot in right_data:
        dt_obj = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
        times_right.append(dt_obj)
        values_right.append(slot['total'] * 100.0)
    n_right = len(values_right)

    # Gemeinsame Skala berechnen aus beiden Panels:
    all_values = values_left + values_right
    if not all_values:
        global_min, global_max = 0.0, 1.0
    else:
        global_min = min(all_values) - 0.5
        global_max = max(all_values) + 0.5
        if global_max <= global_min:
            global_max = global_min + 1.0
    global_range = global_max - global_min
    scale_y = chart_height / global_range

    left_min = global_min
    right_min = global_min
    scale_y_left = scale_y
    scale_y_right = scale_y

    # Zeichne gemeinsame Y-Achse links
    draw.line((chart_x_start, chart_y_top, chart_x_start, chart_y_bottom), fill=0, width=2)
    # Hilfsraster (alle 5ct)
    step = 5
    current_val = left_min - (left_min % step)
    if current_val < 0:
        current_val = 0
    while current_val <= global_max:
        y = chart_y_bottom - (current_val - left_min) * scale_y_left
        draw.line((chart_x_start - 5, y, chart_x_start, y), fill=0, width=1)
        draw.text((chart_x_start - 45, y - 7), f"{current_val/100:.2f}", font=fonts["small"], fill=0)
        current_val += step
    draw.text((chart_x_start - 45, chart_y_top - 20), "Preis (ct/kWh)", font=fonts["small"], fill=0)

    # X-Positionen im linken Panel berechnen (Stunden-Slots)
    x_positions_left = []
    for i in range(n_left):
        x = chart_x_start + i * (panel_width / max(1, n_left - 1))
        x_positions_left.append(x)

    # Zeichne Step-Chart im linken Panel
    for i in range(n_left - 1):
        x1 = x_positions_left[i]
        x2 = x_positions_left[i+1]
        y1 = chart_y_bottom - (values_left[i] - left_min) * scale_y_left
        y2 = chart_y_bottom - (values_left[i+1] - left_min) * scale_y_left
        # Horizontale Linie
        draw.line((x1, y1, x2, y1), fill=0, width=2)
        # Vertikale Linie am Übergang
        draw.line((x2, y1, x2, y2), fill=0, width=2)

    # Gestrichelte Linien und X-Beschriftung
    for i in range(n_left):
        if i < len(times_left):
            x_pos = x_positions_left[i]
            y_val = chart_y_bottom - (values_left[i] - left_min) * scale_y_left
            if y_val < chart_y_bottom:
                draw_dashed_line(draw, x_pos, chart_y_bottom, x_pos, y_val, fill=0, width=1,
                                 dash_length=2, gap_length=2)
            # Stunden-Label (jede 2. Stunde)
            if i % 2 == 0:
                draw.text((x_pos, chart_y_bottom + 5), times_left[i].strftime("%Hh"), font=fonts["small"], fill=0)

    # Tageshoch-/tief im linken Panel
    if n_left > 0:
        lowest_left_index = min(range(n_left), key=lambda i: values_left[i])
        highest_left_index = max(range(n_left), key=lambda i: values_left[i])
        x_low_left = x_positions_left[lowest_left_index]
        y_low_left = chart_y_bottom - (values_left[lowest_left_index] - left_min) * scale_y_left
        x_high_left = x_positions_left[highest_left_index]
        y_high_left = chart_y_bottom - (values_left[highest_left_index] - left_min) * scale_y_left
        draw.text((x_low_left, y_low_left - 15),
                  f"{values_left[lowest_left_index]/100:.2f}", font=fonts["small"], fill=0)
        draw.text((x_high_left, y_high_left - 15),
                  f"{values_left[highest_left_index]/100:.2f}", font=fonts["small"], fill=0)

    # Marker im Future-Modus: horizontale Interpolation, vertikaler Stufenpreis
    if mode == "future" and n_left > 0:
        now_local = datetime.datetime.now(local_tz)
        x_marker_left, y_marker_left, idx_left = get_stepped_marker_position(
            now_local, times_left, x_positions_left, values_left,
            chart_y_bottom, left_min, scale_y_left
        )
        marker_radius = 5
        draw.ellipse((x_marker_left - marker_radius, y_marker_left - marker_radius,
                      x_marker_left + marker_radius, y_marker_left + marker_radius),
                     fill=0)
        # Aktueller Preis
        stepped_price = values_left[idx_left] if idx_left >= 0 else 0
        draw.text((x_marker_left - 35, y_marker_left - 10),
                  f"{stepped_price/100:.2f}", font=fonts["small"], fill=0)

    # --- Rechtes Panel (Preis morgen bzw. Preis heute im Historical-Modus) ---
    x_positions_right = []
    for i in range(n_right):
        x = chart_x_start + panel_width + i * (panel_width / max(1, n_right - 1))
        x_positions_right.append(x)

    for i in range(n_right - 1):
        x1 = x_positions_right[i]
        x2 = x_positions_right[i+1]
        y1 = chart_y_bottom - (values_right[i] - right_min) * scale_y_right
        y2 = chart_y_bottom - (values_right[i+1] - right_min) * scale_y_right
        draw.line((x1, y1, x2, y1), fill=0, width=2)
        draw.line((x2, y1, x2, y2), fill=0, width=2)

    for i in range(n_right):
        if i < len(times_right):
            x_pos = x_positions_right[i]
            y_val = chart_y_bottom - (values_right[i] - right_min) * scale_y_right
            if y_val < chart_y_bottom:
                draw_dashed_line(draw, x_pos, chart_y_bottom, x_pos, y_val, fill=0, width=1,
                                 dash_length=2, gap_length=2)
            if i % 2 == 0:
                draw.text((x_pos, chart_y_bottom + 5), times_right[i].strftime("%Hh"), font=fonts["small"], fill=0)

    if n_right > 0:
        lowest_right_index = min(range(n_right), key=lambda i: values_right[i])
        highest_right_index = max(range(n_right), key=lambda i: values_right[i])
        x_low_right = x_positions_right[lowest_right_index]
        y_low_right = chart_y_bottom - (values_right[lowest_right_index] - right_min) * scale_y_right
        x_high_right = x_positions_right[highest_right_index]
        y_high_right = chart_y_bottom - (values_right[highest_right_index] - right_min) * scale_y_right
        draw.text((x_low_right, y_low_right - 15),
                  f"{values_right[lowest_right_index]/100:.2f}", font=fonts["small"], fill=0)
        draw.text((x_high_right, y_high_right - 15),
                  f"{values_right[highest_right_index]/100:.2f}", font=fonts["small"], fill=0)

    # Marker im Historical-Modus (rechts)
    if mode == "historical" and n_right > 0:
        now_local = datetime.datetime.now(local_tz)
        x_marker_right, y_marker_right, idx_right = get_stepped_marker_position(
            now_local, times_right, x_positions_right, values_right,
            chart_y_bottom, right_min, scale_y_right
        )
        marker_radius = 5
        draw.ellipse((x_marker_right - marker_radius, y_marker_right - marker_radius,
                      x_marker_right + marker_radius, y_marker_right + marker_radius),
                     fill=0)
        stepped_price = values_right[idx_right] if idx_right >= 0 else 0
        draw.text((x_marker_right - 35, y_marker_right - 10),
                  f"{stepped_price/100:.2f}", font=fonts["small"], fill=0)

    # Vertikaler Trenner zwischen den Panels
    x_trenner = chart_x_start + panel_width
    draw.line((x_trenner, chart_y_top, x_trenner, chart_y_bottom), fill=0, width=2)

def draw_subtitle_labels(draw, fonts, mode):
    chart_x_start = 60
    chart_x_end = 800
    panel_width = (chart_x_end - chart_x_start) / 2
    label_y = 415
    bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    if mode == "future":
        draw.text((chart_x_start + 10, label_y), "Preis heute", font=bold_font, fill=0)
        draw.text((chart_x_start + panel_width + 10, label_y), "Preis morgen", font=bold_font, fill=0)
    else:
        draw.text((chart_x_start + 10, label_y), "Verbrauch gestern", font=bold_font, fill=0)
        draw.text((chart_x_start + panel_width + 10, label_y), "Preis heute", font=bold_font, fill=0)

def draw_info_box(draw, data, fonts):
    chart_x_start = 60
    chart_x_end = 800
    info_y = 440
    bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    info_texts = [
        f"Aktueller Preis: {data['current_price']/100:.2f}",
        f"Tagestief: {data['lowest_today']/100:.2f}",
        f"Tageshoch: {data['highest_today']/100:.2f}",
        f"Tiefstpreis in: {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    num_texts = len(info_texts)
    available_width = chart_x_end - chart_x_start
    spacing = available_width / num_texts
    for i, text in enumerate(info_texts):
        x_text = chart_x_start + i * spacing + 5
        draw.text((x_text, info_y), text, font=bold_font, fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init()
   # epd.Clear()

    Himage = Image.new('1', (epd.width, epd.height), 255)
    draw = ImageDraw.Draw(Himage)

    font_small = ImageFont.load_default()
    font_big = ImageFont.load_default()
    info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    fonts = {
        "small": font_small,
        "big": font_big,
        "info_font": info_font
    }

    price_data = get_price_data()
    update_price_cache(price_data)
    cached_yesterday = get_cached_yesterday_price()

    # Modus festlegen: Future-Modus, wenn morgen-Daten vorhanden sind.
    if price_data['tomorrow'] and price_data['tomorrow'][0]['total'] is not None:
        mode = 'future'
    else:
        mode = 'historical'

    if mode == 'future':
        left_data = price_data['today']      # Preis heute
        right_data = price_data['tomorrow']  # Preis morgen
        left_type = "price"
        right_type = "price"
    else:
        if cached_yesterday and cached_yesterday.get('data'):
            # Kombiniere gestrige Preis-Daten und Verbrauchsdaten in einem Dictionary
            consumption_data = get_consumption_data()
            filtered_consumption = filter_yesterday_consumption(consumption_data)
            left_data = {"price": cached_yesterday["data"], "consumption": filtered_consumption}
            left_type = "combo"
        else:
            consumption_data = get_consumption_data()
            left_data = filter_yesterday_consumption(consumption_data)
            left_type = "consumption"
        right_data = price_data['today']
        right_type = "price"

    draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode)
    draw_subtitle_labels(draw, fonts, mode)
    data = prepare_data(price_data)
    draw_info_box(draw, data, fonts)
    draw.text((10, 470), time.strftime("Update: %H:%M %d.%m.%Y"), font=font_small, fill=0)
    epd.display(epd.getbuffer(Himage))
    epd.sleep()

if __name__ == "__main__":
    main()
