#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, time, math, json, requests, datetime, sqlite3
from PIL import Image, ImageDraw, ImageFont

# Zeitzone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
local_tz = ZoneInfo("Europe/Berlin")

# Waveshare-Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2

import api_key

# ---- Tibber-Preis-Cache & Abfrage ----
CACHE_TODAY     = 'cached_today_price.json'
CACHE_YESTERDAY = 'cached_yesterday_price.json'

def save_cache(data, fn):
    with open(fn,'w') as f: json.dump(data,f)
def load_cache(fn):
    if os.path.exists(fn):
        with open(fn) as f: return json.load(f)
    return None

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,
           "Content-Type":"application/json"}
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

def update_price_cache(pd):
    today = datetime.date.today().isoformat()
    ct = load_cache(CACHE_TODAY)
    if not ct or ct.get('date')!=today:
        if ct: save_cache(ct, CACHE_YESTERDAY)
        save_cache({"date":today,"data":pd['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YESTERDAY)

def prepare_data(pd):
    today_vals=[s['total']*100 for s in pd['today']]
    lowest=min(today_vals) if today_vals else 0
    highest=max(today_vals) if today_vals else 0
    cur_dt=datetime.datetime.fromisoformat(pd['current']['startsAt']).astimezone(local_tz)
    cur_price=pd['current']['total']*100
    slots=[(datetime.datetime.fromisoformat(s['startsAt']).astimezone(local_tz), s['total']*100)
           for s in pd['today']+pd['tomorrow']]
    future=[(dt,val) for dt,val in slots if dt>=cur_dt]
    if future:
        ft,fv=min(future,key=lambda x:x[1])
        hours=round((ft-cur_dt).total_seconds()/3600)
    else:
        hours,fv=0,0
    return {"current_price":cur_price,
            "lowest_today":lowest,
            "highest_today":highest,
            "hours_to_lowest":hours,
            "lowest_future_val":fv}

# ---- Preis-Chart ----
def draw_dashed_line(d,x1,y1,x2,y2,**kw):
    dx,dy=x2-x1,y2-y1;dist=math.hypot(dx,dy)
    if dist==0:return
    dl,gl=kw.get('dash_length',4),kw.get('gap_length',4)
    step=dl+gl
    for i in range(int(dist/step)+1):
        s,e=i*step,min(i*step+dl,dist)
        rs, re=s/dist,e/dist
        xa,ya=x1+dx*rs,y1+dy*rs;xb,yb=x1+dx*re,y1+dy*re
        d.line((xa,ya,xb,yb),fill=kw.get('fill',0),width=kw.get('width',1))

def draw_two_day_chart(d,left,right,rt,rt2,fonts,mode,area):pass  # existing implementation

def draw_subtitle_labels(d,fonts,mode):pass

def draw_info_box(d,info,fonts):pass

# ---- PV-Chart (pv1,pv2,dtu_power) ----
DB_FILE='/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db'

def draw_two_day_pv(d,fonts,area):
    import pandas as pd
    X0,Y0,X1,Y1=area;W=X1-X0;H=Y1-Y0;PW=W/2
    def load_day(day):
        st=int(datetime.datetime.combine(day,datetime.time.min).timestamp())
        en=int(datetime.datetime.combine(day,datetime.time.max).timestamp())
        conn=sqlite3.connect(DB_FILE)
        df=pd.read_sql_query(
          'SELECT ts,pv1_power,pv2_power,dtu_power FROM pv_log WHERE ts BETWEEN ? AND ?',
          conn,params=(st,en))
        conn.close()
        df['ts']=pd.to_datetime(df['ts'],unit='s',errors='coerce')
        df.set_index('ts',inplace=True)
        df=df.resample('15T').mean()
        if df.empty:
            idx=pd.date_range(start=datetime.datetime.combine(day,datetime.time.min),periods=96,freq='15T')
            df=pd.DataFrame(index=idx,columns=['pv1_power','pv2_power','dtu_power'])
        df=df.fillna(0)
        return df
    today=datetime.date.today();yday=today-datetime.timedelta(days=1)
    df_y,df_t=load_day(yday),load_day(today)
    vmax=max(df_y['dtu_power'].max(),df_t['dtu_power'].max(),0)+20
    for i,df in enumerate([df_y,df_t]):
        ox=X0+i*PW;n=len(df)
        for col,(dash,wid) in [('pv1_power',(None,2)),('pv2_power',(2,1)),('dtu_power',(None,1))]:
            pts=[]
            for j,val in enumerate(df[col].tolist()):
                if math.isnan(val): val=0
                x=ox+(j*(PW/(n-1)) if n>1 else PW/2)
                y=Y1-int((val/vmax)*H)
                pts.append((x,y))
            for a,b in zip(pts,pts[1:]):
                if dash: draw_dashed_line(d,a[0],a[1],b[0],b[1],dash_length=dash,gap_length=dash,fill=0,width=wid)
                else: d.line((a,b),fill=0,width=wid)
        for h in range(0,25,2):
            x=ox+(h/24)*PW;d.line((x,Y1,x,Y1+4),fill=0);d.text((x-12,Y1+6),f"{h:02d}h",font=fonts['small'],fill=0)
        for v in [0,vmax/2,vmax]:
            y=Y1-int((v/vmax)*H);d.line((ox-5,y,ox,y),fill=0)
            lx=X0-45 if i==0 else X0+PW-45;d.text((lx,y-7),f"{int(v)}W",font=fonts['small'],fill=0)
    d.line((X0+PW,Y0,X0+PW,Y1),fill=0,width=2)

# ---- Main ----
def main():
    epd=epd7in5_V2.EPD();epd.init();epd.Clear()
    img=Image.new('1',(epd.width,epd.height),255);d=ImageDraw.Draw(img)
    fonts={'small':ImageFont.load_default(),'info_font':ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)}
    pd_data=get_price_data();update_price_cache(pd_data)
    cy=get_cached_yesterday();info=prepare_data(pd_data)
    # Preis oben
    upper=(0,0,epd.width,epd.height//2)
    draw_two_day_chart(d,pd_data['today'],pd_data['tomorrow'],None,None,fonts,'future',area=upper)
    draw_subtitle_labels(d,fonts,'future');draw_info_box(d,info,fonts)
    # PV unten
    lower=(0,epd.height//2,epd.width,epd.height)
    draw_two_day_pv(d,fonts,area=lower)
    # Footer
    now=datetime.datetime.now(local_tz).strftime("Update: %H:%M %d.%m.%Y")
    d.text((10,epd.height-20),now,font=fonts['small'],fill=0)
    epd.display(epd.getbuffer(img));epd.sleep();time.sleep(30)
if __name__=='__main__':main()
