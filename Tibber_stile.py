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

# e-Paper Treiber
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib')
sys.path.append('/home/alex/E-Paper-tibber-Preisanzeige/e-paper/lib/waveshare_epd')
from waveshare_epd import epd7in5_V2
import api_key

CACHE_TODAY = 'cached_today_price.json'
CACHE_YEST = 'cached_yesterday_price.json'

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
        if ct: save_cache(ct, CACHE_YEST)
        save_cache({"date":today,"data":pd['today']}, CACHE_TODAY)

def get_cached_yesterday():
    return load_cache(CACHE_YEST)

def get_price_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes {
       currentSubscription { priceInfo {
         today { total startsAt }
         tomorrow { total startsAt }
         current  { total startsAt }
    }}}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",json={"query":q},headers=hdr)
    return r.json()['data']['viewer']['homes'][0]['currentSubscription']['priceInfo']

def get_consumption_data():
    hdr = {"Authorization":"Bearer "+api_key.API_KEY,"Content-Type":"application/json"}
    q = """
    { viewer { homes {
      consumption(resolution:HOURLY,last:48) {
        nodes { from consumption cost }
    }}}}
    """
    r = requests.post("https://api.tibber.com/v1-beta/gql",json={"query":q},headers=hdr)
    j = r.json()
    if "data" not in j or "errors" in j: return None
    return j['data']['viewer']['homes'][0]['consumption']['nodes']

def filter_yesterday_conso(cons):
    yd = datetime.date.today() - datetime.timedelta(days=1)
    out=[]
    if not cons: return out
    for rec in cons:
        try:
            dt = datetime.datetime.fromisoformat(rec['from']).astimezone(local_tz).date()
            if dt==yd: out.append(rec)
        except: pass
    return out

def draw_dashed_line(draw,x1,y1,x2,y2,fill=0,width=1,dash_length=4,gap_length=4):
    dx,dy=x2-x1,y2-y1
    dist=math.hypot(dx,dy)
    if dist==0: return
    per=dash_length+gap_length
    for i in range(int(dist/per)+1):
        s=i*per; e=min(s+dash_length,dist)
        rs, re = s/dist, e/dist
        xa,ya = x1+dx*rs, y1+dy*rs
        xb,yb = x1+dx*re, y1+dy*re
        draw.line((xa,ya,xb,yb),fill=fill,width=width)

def draw_two_day_chart(draw,left_data,left_type,right_data,right_type,fonts,mode):
    # Chart-bounds
    X0, X1 = 60, 800
    Y0, Y1 = 50, 400
    W, H = X1-X0, Y1-Y0
    PW = W/2

    # --- LEFT PANEL DATA ---
    tl,vl = [],[]
    if left_type=="combo":
        # prices yesterday
        for slot in left_data["price"]:
            dt = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            tl.append(dt); vl.append(slot['total']*100)
        # consumption
        cl = [rec['consumption'] for rec in left_data['consumption']]
    elif left_type=="price":
        for slot in left_data:
            dt = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
            tl.append(dt); vl.append(slot['total']*100)
        cl=[]
    else:  # pure consumption
        for slot in left_data:
            dt = datetime.datetime.fromisoformat(slot['from']).astimezone(local_tz)
            tl.append(dt); vl.append(slot['consumption'])
        cl=[]

    # --- RIGHT PANEL (always price) ---
    tr,vr = [],[]
    for slot in right_data:
        dt = datetime.datetime.fromisoformat(slot['startsAt']).astimezone(local_tz)
        tr.append(dt); vr.append(slot['total']*100)

    # y-scale
    allv = vl+vr
    gm = min(allv)-0.5 if allv else 0
    gM = max(allv)+0.5 if allv else 1
    rng = gM-gm
    sy = H/rng

    # draw axes & y-labels
    draw.line((X0,Y0,X0,Y1),fill=0,width=2)
    step=5
    yval = 0
    while yval<=gM:
        y = Y1 - (yval-gm)*sy
        draw.line((X0-5,y,X0,y),fill=0)
        draw.text((X0-45,y-7),f"{yval/100:.2f}",font=fonts["small"],fill=0)
        yval+=step
    draw.text((X0-45,Y0-20),"Preis (ct/kWh)",font=fonts["small"],fill=0)

    # x-positions left
    nL=len(vl)
    xl = [X0 + i*(PW/(nL-1)) for i in range(nL)] if nL>1 else [X0]

    # price line left
    for i in range(nL-1):
        x1,y1 = xl[i], Y1-(vl[i]-gm)*sy
        x2,y2 = xl[i+1], Y1-(vl[i+1]-gm)*sy
        draw.line((x1,y1,x2,y1),fill=0,width=2)
        draw.line((x2,y1,x2,y2),fill=0,width=2)
    # consumption dashed line in historical/combo
    if mode=="historical" and left_type=="combo" and cl:
        # scale consumption to same range?
        maxc=max(cl); minc=min(cl)
        # map each consumption to y
        yc = [Y1 - ((c-minc)/(maxc-minc))*H for c in cl]
        for i in range(len(yc)-1):
            draw_dashed_line(draw, xl[i], yc[i], xl[i+1], yc[i+1], fill=0, width=1, dash_length=3, gap_length=3)

    # x-labels left
    for i,dt in enumerate(tl):
        if i%2==0:
            draw.text((xl[i],Y1+5),dt.strftime("%Hh"),font=fonts["small"],fill=0)

    # lowest-price marker & price text
    if vl:
        idx_low = min(range(nL), key=lambda i:vl[i])
        xL,yL = xl[idx_low], Y1-(vl[idx_low]-gm)*sy
        r=5
        draw.ellipse((xL-r,yL-r,xL+r,yL+r),fill=0)
        # price
        draw.text((xL,yL-15),f"{vl[idx_low]/100:.2f}",font=fonts["small"],fill=0)
        # ** hour label ~20px above **
        hour_txt = tl[idx_low].strftime("%Hh")
        h_w,h_h = fonts["info_font"].getsize(hour_txt)
        draw.text((xL - h_w/2, yL - 15 - 20), hour_txt, font=fonts["info_font"], fill=0)

    # --- RIGHT PANEL --- similar ...
    # omitted for brevity; keep as in original for right side price plotting

    # separator
    draw.line((X0+PW,Y0,X0+PW,Y1),fill=0,width=2)

    # return left chart params if needed...
    return

def draw_info_box(draw,data,fonts):
    X0,X1=60,800
    Y=440
    bold = fonts["info_font"]
    texts = [
      f"Aktueller Preis: {data['current_price']/100:.2f}",
      f"Tagestief:      {data['lowest_today']/100:.2f}",
      f"Tageshoch:      {data['highest_today']/100:.2f}",
      f"Tiefstpreis in: {data['hours_to_lowest']}h | {data['lowest_future_val']/100:.2f}"
    ]
    w = (X1-X0)/len(texts)
    for i,t in enumerate(texts):
        draw.text((X0 + i*w + 5, Y), t, font=bold, fill=0)

def main():
    epd = epd7in5_V2.EPD()
    epd.init(); epd.Clear()

    img = Image.new('1',(epd.width,epd.height),255)
    d = ImageDraw.Draw(img)
    f_small = ImageFont.load_default()
    f_info  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    fonts={"small":f_small,"info_font":f_info}

    pd = get_price_data()
    update_price_cache(pd)
    cy = get_cached_yesterday()
    if pd['tomorrow'] and pd['tomorrow'][0]['total'] is not None:
        mode='future'
        ldata, ltype = pd['today'], "price"
        rdata, rtype = pd['tomorrow'], "price"
    else:
        mode='historical'
        if cy and cy.get('data'):
            cons = filter_yesterday_conso(get_consumption_data())
            ldata={"price":cy['data'],"consumption":cons}
            ltype="combo"
        else:
            cons = filter_yesterday_conso(get_consumption_data())
            ldata=cons; ltype="consumption"
        rdata, rtype = pd['today'], "price"

    draw_two_day_chart(d,ldata,ltype,rdata,rtype,fonts,mode)
    data = { 
      "current_price": pd['current']['total']*100,
      "lowest_today": min([s['total']*100 for s in pd['today']]),
      "highest_today": max([s['total']*100 for s in pd['today']]),
      "hours_to_lowest": 0, "lowest_future_val":0
    }
    draw_info_box(d,data,fonts)
    d.text((10,470), time.strftime("Update: %H:%M %d.%m.%Y"), font=fonts["small"], fill=0)
    epd.display(epd.getbuffer(img))
    epd.sleep()

if __name__=="__main__":
    main()
    time.sleep(30)