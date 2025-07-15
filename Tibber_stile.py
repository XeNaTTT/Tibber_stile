#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import math
import json
import requests
import datetime
import sqlite3
import logging

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# Logging
LOG_FILE = '/home/alex/logs/tibber.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# E-Paper Pfade
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key
DB_FILE = '/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'

CACHE_TODAY     = '/home/alex/E-Paper-tibber-Preisanzeige/cached_today_price.json'
CACHE_YESTERDAY = '/home/alex/E-Paper-tibber-Preisanzeige/cached_yesterday_price.json'


def save_cache(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)


def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return None


def get_price_data():
    hdr = {"Authorization": f"Bearer {api_key.API_KEY}", "Content-Type": "application/json"}
    query = '''
    { viewer { homes { currentSubscription { priceInfo {
      today    { total startsAt }
      tomorrow { total startsAt }
      current  { total startsAt }
    }}}}}
    '''
    try:
        r = requests.post("https://api.tibber.com/v1-beta/gql", json={"query": query}, headers=hdr, timeout=20)
        r.raise_for_status()
        return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']
    except Exception as e:
        logging.error(f"API-Fehler: {e}")
        return None


def update_price_cache(pi):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date') != today:
        if ct:
            save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date": today, "data": pi.get('today', [])}, CACHE_TODAY)


def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY) or {"data": []}


def prepare_data(pi):
    # Berechnet Statistiken
    today_list = pi.get('today', [])
    vals = [s['total']*100 for s in today_list]
    lowest = min(vals) if vals else 0
    highest = max(vals) if vals else 0

    cur = pi.get('current', {})
    try:
        cur_dt = datetime.datetime.fromisoformat(cur['startsAt']).astimezone(local_tz)
        cur_price = cur['total']*100
    except:
        cur_dt = datetime.datetime.now(local_tz)
        cur_price=0

    # Zukunftsplanung
    slots = []
    for seg in pi.get('today', []) + pi.get('tomorrow', []):
        try:
            dt = datetime.datetime.fromisoformat(seg['startsAt']).astimezone(local_tz)
            slots.append((dt, seg['total']*100))
        except:
            pass
    future = [(dt, v) for dt, v in slots if dt>=cur_dt]
    if future:
        ft, fv = min(future, key=lambda x: x[1])
        hours = round((ft-cur_dt).total_seconds()/3600)
    else:
        hours, fv = 0,0

    return {"current_price":cur_price,
            "lowest_today":lowest,
            "highest_today":highest,
            "hours_to_lowest":hours,
            "lowest_future_val":fv}


def get_pv_series(slots):
    conn=sqlite3.connect(DB_FILE)
    df=pd.read_sql_query("SELECT ts,dtu_power FROM pv_log", conn)
    conn.close()
    df['ts']=pd.to_datetime(df['ts'],unit='s')
    df.set_index('ts',inplace=True)
    df=df.resample('15T').mean().ffill().fillna(0)
    df.index=df.index.tz_localize(local_tz)
    last=df.index.max()
    vals=[]
    for s in slots:
        t=s['startsAt']
        dt=t if isinstance(t,datetime.datetime) else datetime.datetime.fromisoformat(t).astimezone(local_tz)
        if dt<=last:
            v=df['dtu_power'].asof(dt)
            vals.append(float(v) if not pd.isna(v) else 0)
        else:
            vals.append(np.nan)
    return pd.Series(vals)


def draw_dashed_line(d,x1,y1,x2,y2,**kw):
    dx,dy=x2-x1,y2-y1
    dist=math.hypot(dx,dy)
    if dist==0: return
    dl,gl=kw.get('dash_length',4),kw.get('gap_length',4)
    step=dl+gl
    for i in range(int(dist/step)+1):
        s=i*step; e=min(s+dl,dist)
        rs, re=s/dist,e/dist
        xa,ya=x1+dx*rs,y1+dy*rs
        xb,yb=x1+dx*re,y1+dy*re
        d.line((xa,ya,xb,yb),fill=kw.get('fill',0),width=kw.get('width',1))


def draw_two_day_chart(d,left,right,fonts,subtitles,area,pv_y=None,pv_t=None,label_min_max=False):
    X0,Y0,X1,Y1=area
    W,H=X1-X0,Y1-Y0; PW=W/2
    # Daten extrahieren
    def ext(slots):
        ts,ps=[],[]
        for s in slots:
            if isinstance(s,dict): t,p=s['startsAt'],s['total']*100
            elif isinstance(s,tuple): t,p=s
            else: continue
            try: dt=t if isinstance(t,datetime.datetime) else datetime.datetime.fromisoformat(t).astimezone(local_tz)
            except: continue
            ts.append(dt); ps.append(p)
        return ts,ps
    tl,vl=ext(left); tr,vr=ext(right)
    allp=vl+vr
    if not allp: return
    vmin,vmax=min(allp)-0.5,max(allp)+0.5; sy=H/(vmax-vmin)
    # PV-Skala
    syv=None
    pmax=0
    if pv_y is not None: pmax=max(pmax,pv_y.max())
    if pv_t is not None: pmax=max(pmax,pv_t.max())
    if pmax>0: syv=H/(pmax+20)
    # Y-Achse
    step=5; yv=math.floor(vmin/step)*step
    while yv<=vmax:
        y=Y1-(yv-vmin)*sy
        d.line((X0-5,y,X0,y),fill=0); d.line((X1,y,X1+5,y),fill=0)
        d.text((X0-45,y-7),f"{yv/100:.2f}",font=fonts['small'],fill=0)
        yv+=step
    d.text((X0-45,Y0-20),'Preis (ct/kWh)',font=fonts['small'],fill=0)
    # Untertitel
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    d.text((X0+5,Y1+5),subtitles[0],font=bf,fill=0)
    d.text((X0+PW+5,Y1+5),subtitles[1],font=bf,fill=0)
    # Panel
    def panel(ts,ps,pv,x0):
        n=len(ts)
        if n<2: return
        xs=[x0+i*(PW/(n-1)) for i in range(n)]
        # Preis
        for i in range(n-1):
            x1,y1=xs[i],Y1-(ps[i]-vmin)*sy; x2,y2=xs[i+1],Y1-(ps[i+1]-vmin)*sy
            d.line((x1,y1,x2,y1),fill=0,width=2); d.line((x2,y1,x2,y2),fill=0,width=2)
        # Label
        if label_min_max:
            im,ix=ps.index(min(ps)),ps.index(max(ps))
            for idx in (im,ix):
                xi,yi=xs[idx],Y1-(ps[idx]-vmin)*sy
                d.text((xi-12,yi-12),f"{ps[idx]/100:.2f}",font=fonts['small'],fill=0)
        else:
            for i in range(n-1): mx=(xs[i]+xs[i+1])/2; yv=Y1-(ps[i]-vmin)*sy; d.text((mx-12,yv-12),f"{ps[i]/100:.2f}",font=fonts['small'],fill=0)
        # PV
        if syv and pv is not None:
            pts=[]
            for i in range(n): pts.append((xs[i],Y1-int(pv.iloc[i]*syv)) if not np.isnan(pv.iloc[i]) else None)
            for a,b in zip(pts,pts[1:]):
                if a and b: draw_dashed_line(d,a[0],a[1],b[0],b[1],dash_length=2,gap_length=2)
            valid=[i for i in range(n) if pts[i]]
            if valid:
                imax=valid[np.nanargmax([pv.iloc[i] for i in valid])]
                x,y=pts[imax]; d.text((x-15,y-15),f"{int(pv.iloc[imax])}W",font=fonts['small'],fill=0)
        # Zeit
        for i,dt in enumerate(ts):
            if i%2==0: d.text((xs[i],Y1+18),dt.strftime('%Hh'),font=fonts['small'],fill=0)
    panel(tl,vl,pv_y,X0)
    d.line((X0+PW,Y0,X0+PW,Y1),fill=0,width=2)
    panel(tr,vr,pv_t,X0+PW)


def draw_info_box(d,info,fonts,y):
    X0,X1=60,epd7in5_V2.EPD().width
    bf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
    texts=[
        f"Preis jetzt: {info['current_price']/100:.2f}",
        f"Tagestief:   {info['lowest_today']/100:.2f}",
        f"Tageshoch:   {info['highest_today']/100:.2f}",
        f"Tiefst in {info['hours_to_lowest']}h"
    ]
    w=(X1-X0)/len(texts)
    for i,t in enumerate(texts): d.text((X0+i*w+5,y),t,font=bf,fill=0)


def main():
    epd=epd7in5_V2.EPD();epd.init();epd.Clear()
    pi=get_price_data()
    if not pi: logging.error("Keine Preisdaten"); return
    update_price_cache(pi)
    yest,td,tm= get_cached_yesterday().get('data',[]), pi.get('today',[]), pi.get('tomorrow',[])
    w,h=epd.width,epd.height; mx,my=int(w*0.05),int(h*0.05)
    area=(mx,my,w-mx,h-30)
    if tm:
        left,right=td,tm; pv_l,pv_r=get_pv_series(td),None
        subs=("Preis & PV heute","Preis & PV morgen"); lm=True
    else:
        left,right=yest,td; pv_l,pv_r=get_pv_series(yest),get_pv_series(td)
        subs=("Preise & PV gestern","Preis & PV heute"); lm=False
    info=prepare_data(pi)
    img=Image.new('1',(w,h),255); d=ImageDraw.Draw(img)
    fonts={'small':ImageFont.load_default(),'info_font':ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)}
    draw_two_day_chart(d,left,right,fonts,subs,area,pv_y=pv_l,pv_t=pv_r,label_min_max=lm)
    info_y=area[3]+5
    draw_info_box(d,info,fonts,info_y)
    d.text((10,h-20),datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y"),font=fonts['small'],fill=0)
    epd.display(epd.getbuffer(img));epd.sleep();time.sleep(30)

if __name__=="__main__":
    try: main()
    except: logging.exception("Fehler")
