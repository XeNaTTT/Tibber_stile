#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys, os, time, math, json, requests, datetime
from PIL import Image, ImageDraw, ImageFont

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# E-Paper Treiberpfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2
import api_key

# Caching
CACHE_TODAY     = 'cached_today_price.json'
CACHE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data,f)
def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date')!=today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY)

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query":q}, headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def get_consumption_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes { consumption(resolution:HOURLY,last:48) {
      nodes { from consumption cost }
    }}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query":q}, headers=hdr)
    j = r.json()
    if "data" not in j or "errors" in j: return None
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_consumption(cons):
    yd = datetime.date.today() - datetime.timedelta(days=1)
    out=[]
    if not cons: return out
    for r in cons:
        try:
            dt = datetime.datetime.fromisoformat(r['from']).astimezone(local_tz).date()
            if dt==yd: out.append(r)
        except: pass
    return out

def draw_dashed_line(draw, x1,y1,x2,y2, fill=0, width=1, dash_length=4, gap_length=4):
    dx,dy = x2-x1, y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    step = dash_length+gap_length
    for i in range(int(dist/step)+1):
        s=i*step; e=min(s+dash_length,dist)
        rs, re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        draw.line((xa,ya,xb,yb), fill=fill, width=width)

def draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode):
    # Chart‐Region
    X0,X1 = 60,800
    Y0,Y1 = 50,400
    W,H   = X1-X0, Y1-Y0
    PW    = W/2

    # Gemeinsame Y‐Skalierung
    vals_l = [slot['total']*100 for slot in left_data]
    vals_r = [slot['total']*100 for slot in right_data]
    allv = vals_l + vals_r
    vmin, vmax = (min(allv)-0.5, max(allv)+0.5) if allv else (0,1)
    sy = H/(vmax-vmin)

    # Y-Achse und Ticks
    step = 5
    yv = math.floor(vmin/step)*step
    while yv <= vmax:
        y = Y1 - (yv-vmin)*sy
        # links und rechts Tick
        draw.line((X0-5,y, X0,y), fill=0)
        draw.line((X1,y, X1+5,y), fill=0)
        draw.text((X0-45,y-7), f"{yv/100:.2f}", font=fonts['small'], fill=0)
        yv += step
    draw.text((X0-45,Y0-20),"Preis (ct/kWh)", font=fonts['small'], fill=0)

    # Linkes Panel: Preise heute/gestern
    times_l = [datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz) for s in left_data]
    xL = [X0 + i*(PW/(len(times_l)-1)) for i in range(len(times_l))] if len(times_l)>1 else [X0]
    for i in range(len(times_l)-1):
        x1,y1 = xL[i],   Y1-(vals_l[i]-vmin)*sy
        x2,y2 = xL[i+1], Y1-(vals_l[i+1]-vmin)*sy
        draw.line((x1,y1,x2,y1), fill=0, width=2)
        draw.line((x2,y1,x2,y2), fill=0, width=2)

    # Verbrauch im historical-Modus
    if mode=='historical':
        cons_nodes = filter_yesterday_consumption(get_consumption_data())
        if cons_nodes and len(cons_nodes)==len(times_l):
            cons = [r['consumption'] for r in cons_nodes]
            cmin,cmax = min(cons), max(cons)
            span = cmax-cmin if cmax>cmin else 1
            y_cons = [Y1 - ((c-cmin)/span)*H for c in cons]
            for i in range(len(y_cons)-1):
                draw_dashed_line(draw,
                    xL[i],   y_cons[i],
                    xL[i+1], y_cons[i+1],
                    fill=0, width=1, dash_length=4, gap_length=4)

    # X-Beschriftung links
    for i,dt in enumerate(times_l):
        if i%2==0:
            draw.text((xL[i],Y1+5), dt.strftime("%Hh"), font=fonts['small'], fill=0)

    # Rechtes Panel: Preise morgen/heute
    times_r = [datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz) for s in right_data]
    xR = [X0+PW + i*(PW/(len(times_r)-1)) for i in range(len(times_r))] if len(times_r)>1 else [X0+PW]
    for i in range(len(times_r)-1):
        x1,y1 = xR[i],   Y1-(vals_r[i]-vmin)*sy
        x2,y2 = xR[i+1], Y1-(vals_r[i+1]-vmin)*sy
        draw.line((x1,y1,x2,y1), fill=0, width=2)
        draw.line((x2,y1,x2,y2), fill=0, width=2)
    # X-Beschriftung rechts
    for i,dt in enumerate(times_r):
        if i%2==0:
            draw.text((xR[i],Y1+5), dt.strftime("%Hh"), font=fonts['small'], fill=0)

    # Trenner
    draw.line((X0+PW,Y0,X0+PW,Y1), fill=0, width=2)

    # Aktueller Marker
    now = datetime.datetime.now(local_tz)
    def locate(times, vals, xs):
        for i in range(len(times)-1):
            if times[i] <= now < times[i+1]:
                frac = (now-times[i]).total_seconds()/(times[i+1]-times[i]).total_seconds()
                return xs[i] + frac*(xs[i+1]-xs[i]), Y1-(vals[i]-vmin)*sy, i
        return xs[-1], Y1-(vals[-1]-vmin)*sy, len(vals)-1

    if mode=='future':
        xm,ym,idx = locate(times_l, vals_l, xL)
    else:
        xm,ym,idx = locate(times_r, vals_r, xR)

    r=5
    draw.ellipse((xm-r,ym-r,xm+r,ym+r), fill=0)
    price = (vals_l if mode=='future' else vals_r)[idx]/100
    draw.text((xm+8,ym-8), f"{price:.2f}", font=fonts['small'], fill=0)

def draw_subtitle_labels(draw,fonts,mode):
    X0,X1=60,800; PW=(X1-X0)/2; y=415
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    if mode=="future":
        draw.text((X0+10,y),"Preis heute", font=bf, fill=0)
        draw.text((X0+PW+10,y),"Preis morgen", font=bf, fill=0)
    else:
        draw.text((X0+10,y),"Preise gestern", font=bf, fill=0)
        draw.text((X0+PW+10,y),"Preis heute", font=bf, fill=0)

def draw_info_box(draw,data,fonts):
    X0,X1=60,800; y=440
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    texts=[
        f"Aktueller Preis: {data['current_price']/100:.2f}",
        f"Tagestief:       {data['lowest_today']/100:.2f}",
        f"Tageshoch:       {data['highest_today']/100:.2f}",
        f"Tiefstpreis in:  {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    w=(X1-X0)/len(texts)
    for i,t in enumerate(texts):
        draw.text((X0+i*w+5,y),t,font=bf,fill=0)

def main():
    epd = epd7in5_V2.EPD(); epd.init(); epd.Clear()
    img = Image.new('1',(epd.width,epd.height),255)
    draw = ImageDraw.Draw(img)
    fonts = {
        "small": ImageFont.load_default(),
        "info_font": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    }

    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday()

    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode='future'
        left_data, left_type = pd['today'], "price"
        right_data, right_type = pd['tomorrow'], "price"
    else:
        mode='historical'
        # gestern aus Cache + Verbrauch
        yd = (get_cached_yesterday() or {}).get('data',[])
        left_data = [
            {"startsAt":s.get('startsAt',s.get('from')),
             "total":   s.get('total',0),
             "consumption": s.get('consumption',0)}
            for s in yd
        ]
        left_type, right_data, right_type = "price", pd['today'], "price"

    draw_two_day_chart(draw, left_data, left_type, right_data, right_type, fonts, mode)
    draw_subtitle_labels(draw, fonts, mode)
    data = {
        "current_price":    pd['current']['total']*100,
        "lowest_today":     min([s['total']*100 for s in pd['today']]),
        "highest_today":    max([s['total']*100 for s in pd['today']]),
        "hours_to_lowest":  0,
        "lowest_future_val":0
    }
    draw_info_box(draw, data, fonts)
    draw.text((10,470), time.strftime("Update: %H:%M %d.%m.%Y"), font=fonts['small'], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__=="__main__":
    main()
    time.sleep(30)