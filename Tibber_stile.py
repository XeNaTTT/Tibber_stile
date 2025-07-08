#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime
from PIL import Image, ImageDraw, ImageFont

# Zeitzone (Python 3.9+ oder backports)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Pfade zum Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

CACHE_FILE_TODAY     = 'cached_today_price.json'
CACHE_FILE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data,f)
def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_FILE_TODAY)
    if not ct or ct.get('date')!=today:
        if ct: save_cache(ct, CACHE_FILE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_FILE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_FILE_YESTERDAY)

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query":q}, headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def get_consumption_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes { consumption(resolution:HOURLY,last:48) {
      nodes { from consumption cost }
    }}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",
                      json={"query":q}, headers=hdr)
    j = r.json()
    if "data" not in j or "errors" in j:
        return None
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_consumption(cons):
    yd = datetime.date.today() - datetime.timedelta(days=1)
    out=[]
    if not cons: return out
    for r in cons:
        try:
            d = datetime.datetime.fromisoformat(r['from']).astimezone(local_tz).date()
            if d==yd: out.append(r)
        except: pass
    return out

def prepare_data(pd):
    # Aktueller Preis, Tief & Hoch heute, nächster Tiefstpreis
    today_vals = [s['total']*100 for s in pd['today']]
    lowest_today  = min(today_vals) if today_vals else 0
    highest_today = max(today_vals) if today_vals else 0
    cur_dt = datetime.datetime.fromisoformat(pd['current']['startsAt']).astimezone(local_tz)
    cur_price = pd['current']['total']*100
    # future
    all_slots = [(datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz),
                  s['total']*100) for s in pd['today']+pd['tomorrow']]
    future = [(dt,val) for dt,val in all_slots if dt>=cur_dt]
    if future:
        ft,fv = min(future, key=lambda x:x[1])
        hours_to_lowest = round((ft-cur_dt).total_seconds()/3600)
        lowest_future_val = fv
    else:
        hours_to_lowest = 0
        lowest_future_val = 0
    return {
        "current_price":    cur_price,
        "lowest_today":     lowest_today,
        "highest_today":    highest_today,
        "hours_to_lowest":  hours_to_lowest,
        "lowest_future_val":lowest_future_val
    }

def draw_dashed_line(d, x1,y1,x2,y2, **kw):
    dx,dy = x2-x1,y2-y1
    dist = math.hypot(dx,dy)
    if dist==0: return
    dl,gl = kw.get("dash_length",4), kw.get("gap_length",4)
    step = dl+gl
    for i in range(int(dist/step)+1):
        s=i*step; e=min(s+dl,dist)
        rs,re = s/dist,e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        d.line((xa,ya,xb,yb), fill=kw.get("fill",0), width=kw.get("width",1))

def draw_two_day_chart(d, left_data, lt, right_data, rt, fonts, mode):
    X0,X1=60,800; Y0,Y1=50,400; W,H=X1-X0,Y1-Y0; PW=W/2
    vals_l = [s['total']*100 for s in left_data]
    vals_r = [s['total']*100 for s in right_data]
    allv=vals_l+vals_r
    if allv:
        vmin,vmax=min(allv)-0.5,max(allv)+0.5
    else:
        vmin,vmax=0,1
    sy=H/(vmax-vmin)

    # Y-Achse mit Ticks und Labels
    step=5; yv=math.floor(vmin/step)*step
    while yv<=vmax:
        y=Y1-(yv-vmin)*sy
        d.line((X0-5,y,X0,y),fill=0)
        d.line((X1,y,X1+5,y),fill=0)
        d.text((X0-45,y-7),f"{yv/100:.2f}",font=fonts["small"],fill=0)
        yv+=step
    d.text((X0-45,Y0-20),"Preis (ct/kWh)",font=fonts["small"],fill=0)

    # ** Neuer Haupt-Achsenstrich links **
    d.line((X0, Y0, X0, Y1), fill=0, width=1)

    # linkes Panel (heute oder gestern)
    times_l=[datetime.datetime.fromisoformat(s["startsAt"]).astimezone(local_tz)
             for s in left_data]
    nL=len(times_l)
    xL=[X0+i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]
    for i in range(nL-1):
        x1,y1=xL[i],Y1-(vals_l[i]-vmin)*sy
        x2,y2=xL[i+1],Y1-(vals_l[i+1]-vmin)*sy
        d.line((x1,y1,x2,y1),fill=0,width=2)
        d.line((x2,y1,x2,y2),fill=0,width=2)
    if mode=="historical":
        cons=filter_yesterday_consumption(get_consumption_data())
        if cons and len(cons)==nL:
            cv=[r["consumption"] for r in cons]
            cmin,cmax=min(cv),max(cv); span=(cmax-cmin if cmax>cmin else 1)
            y_c=[Y1-((c-cmin)/span)*H for c in cv]
            for i in range(len(y_c)-1):
                draw_dashed_line(d,xL[i],y_c[i],xL[i+1],y_c[i+1],fill=0)
    for i,dt in enumerate(times_l):
        if i%2==0:
            d.text((xL[i],Y1+5),dt.strftime("%Hh"),font=fonts["small"],fill=0)

    # rechtes Panel (morgen oder heute)
    times_r=[datetime.datetime.fromisoformat(s["startsAt"]).astimezone(local_tz)
             for s in right_data]
    nR=len(times_r)
    xR=[X0+PW+i*(PW/(nR-1)) for i in range(nR)] if nR>1 else [X0+PW]
    for i in range(nR-1):
        x1,y1=xR[i],Y1-(vals_r[i]-vmin)*sy
        x2,y2=xR[i+1],Y1-(vals_r[i+1]-vmin)*sy
        d.line((x1,y1,x2,y1),fill=0,width=2)
        d.line((x2,y1,x2,y2),fill=0,width=2)
    for i,dt in enumerate(times_r):
        if i%2==0:
            d.text((xR[i],Y1+5),dt.strftime("%Hh"),font=fonts["small"],fill=0)

    # Annotate highs & low heute und — im future-Modus — auch für morgen
    if mode=="future":
        times_today, vals_today, xs = times_l, vals_l, xL
    else:
        times_today, vals_today, xs = times_r, vals_r, xR

    if vals_today:
        # Index des niedrigsten Preises heute
        idx_min = vals_today.index(min(vals_today))
        # heute: niedrigster, höchster, 2. höchster
        sidx = sorted(range(len(vals_today)), key=lambda i: vals_today[i])
        for idx,off in (
            (sidx[0],40),
            (sidx[-1],15),
            (sidx[-2] if len(sidx)>1 else sidx[-1],15)
        ):
            x = xs[idx]
            y = Y1 - (vals_today[idx]-vmin)*sy
            d.text((x-15, y-off), f"{vals_today[idx]/100:.2f}", font=fonts["small"], fill=0)

        # Uhrzeit des niedrigsten Preises heute mittig im Chart
        x0 = xs[idx_min]
        mid_y = (Y0 + Y1) / 2
        txt = times_today[idx_min].strftime("%Hh")
        d.text((x0 - 10, mid_y), txt, font=fonts["info_font"], fill=0)

        # ** NEU: im future-Modus auch morgen annotieren **
        if mode=="future" and vals_r:
            # sortierte Indizes der morgigen Werte
            tomorrow_idxs = sorted(range(len(vals_r)), key=lambda i: vals_r[i])
            # (tiefster, 2. höchster, höchster) mit Offsets
            to_annotate = [
                (tomorrow_idxs[0],    40),
                (tomorrow_idxs[-2] if len(tomorrow_idxs)>1 else tomorrow_idxs[-1], 15),
                (tomorrow_idxs[-1],   15)
            ]
            for idx_r, off in to_annotate:
                xr = xR[idx_r]
                yr = Y1 - (vals_r[idx_r] - vmin) * sy
                d.text(
                    (xr - 15, yr - off),
                    f"{vals_r[idx_r]/100:.2f}",
                    font=fonts["small"],
                    fill=0
                )

    # Trenner Linie in der Mitte
    d.line((X0+PW, Y0, X0+PW, Y1), fill=0, width=2)

    # Marker für die aktuelle Stunde
    now = datetime.datetime.now(local_tz)
    def locate(tms, vs, xs):
        for i in range(len(tms)-1):
            if tms[i] <= now < tms[i+1]:
                frac = (now - tms[i]).total_seconds() / (tms[i+1] - tms[i]).total_seconds()
                return xs[i] + frac*(xs[i+1]-xs[i]), Y1-(vs[i]-vmin)*sy, i
        return xs[-1], Y1-(vs[-1]-vmin)*sy, len(vs)-1

    if mode=="future":
        xm, ym, ix = locate(times_l, vals_l, xL)
    else:
        xm, ym, ix = locate(times_r, vals_r, xR)

    r = 5
    d.ellipse((xm-r, ym-r, xm+r, ym+r), fill=0)
    pr = (vals_l if mode=="future" else vals_r)[ix] / 100
    d.text((xm+8, ym-8), f"{pr:.2f}", font=fonts["small"], fill=0)

def draw_subtitle_labels(d,fonts,mode):
    X0,X1=60,800; PW=(X1-X0)/2; y=415
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    if mode=="future":
        d.text((X0+10,y),"Preis heute",font=bf,fill=0)
        d.text((X0+PW+10,y),"Preis morgen",font=bf,fill=0)
    else:
        d.text((X0+10,y),"Preise gestern",font=bf,fill=0)
        d.text((X0+PW+10,y),"Preis heute",font=bf,fill=0)

def draw_info_box(d,data,fonts):
    X0,X1=60,800; y=440
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    infos=[
        f"Aktueller Preis: {data['current_price']/100:.2f}",
        f"Tagestief:       {data['lowest_today']/100:.2f}",
        f"Tageshoch:       {data['highest_today']/100:.2f}",
        f"Tiefstpreis in:  {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    w=(X1-X0)/len(infos)
    for i,t in enumerate(infos):
        d.text((X0+i*w+5,y),t,font=bf,fill=0)

def main():
    epd=epd7in5_V2.EPD()
    epd.init()
    epd.Clear()

    img=Image.new('1',(epd.width,epd.height),255)
    d=ImageDraw.Draw(img)
    fonts={
        "small":ImageFont.load_default(),
        "info_font":ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    }

    pd=get_price_data()
    update_price_cache(pd)
    cy=get_cached_yesterday()
    info=prepare_data(pd)

    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode='future'
        left_data=pd['today']
        right_data=pd['tomorrow']
    else:
        mode='historical'
        ydata=cy.get('data',[]) if cy else []
        left_data=[{
            "startsAt": r.get('startsAt',r.get('from')),
            "total":    r.get('total',0)
        } for r in ydata]
        right_data=pd['today']

    draw_two_day_chart(d, left_data, "price", right_data, "price", fonts, mode)
    draw_subtitle_labels(d, fonts, mode)
    draw_info_box(d, info, fonts)
    d.text((10,470), time.strftime("Update: %H:%M %d.%m.%Y"), font=fonts["small"], fill=0)

    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__=="__main__":
    main()
    time.sleep(30)