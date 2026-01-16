# battery_peak_shaver.py
# PV-only dynamic peak shaving using EcoFlow STREAM + Tibber prices
# Nutzbare Kapazität: 1920 Wh

BATT_CAPACITY_WH = 1920
SOC_MIN = 15
RESERVE_HYSTERESIS_PCT = 2
HORIZON_HOURS = 18
TOP_PERCENT = 0.25
# --------------------------------------------------------------

import time, json, sqlite3, datetime as dt, requests
import api_key
from urllib.parse import urlencode
import hmac, hashlib

DB = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"
TZ = dt.timezone(dt.timedelta(hours=1))

def clamp_reserve(x):
    try: x = int(round(float(x)))
    except: x = SOC_MIN
    return max(3, min(95, x))

# ---- EcoFlow signing helpers (gekürzt) ----
def _six_digit_nonce(): return f"{int(time.time()*1000)%900000+100000}"

def _flatten_params(obj,p=""):
    it=[]
    if isinstance(obj,dict):
        for k in obj: it+=_flatten_params(obj[k],f"{p}.{k}" if p else k)
    else: it.append((p,str(obj)))
    return it

def _sign(params,ak,sk):
    ts=str(int(time.time()*1000)); nonce=_six_digit_nonce()
    kv=sorted(_flatten_params(params))
    base="&".join(f"{k}={v}" for k,v in kv)
    msg=(base+"&" if base else "")+f"accessKey={ak}&nonce={nonce}&timestamp={ts}"
    sig=hmac.new(sk.encode(),msg.encode(),hashlib.sha256).hexdigest()
    return {"accessKey":ak,"nonce":nonce,"timestamp":ts,"sign":sig}

HOST=getattr(api_key,"ECOFLOW_HOST","https://api-e.ecoflow.com").rstrip("/")

def eco_get_main_sn(sn):
    url=f"{HOST}/iot-open/sign/device/system/main/sn?{urlencode({'sn':sn})}"
    r=requests.get(url,headers=_sign({"sn":sn},api_key.ECOFLOW_APP_KEY,api_key.ECOFLOW_SECRET_KEY),timeout=10)
    return r.json().get("data",{}).get("sn",sn)

def eco_quota(sn):
    url=f"{HOST}/iot-open/sign/device/quota/all?{urlencode({'sn':sn})}"
    r=requests.get(url,headers=_sign({"sn":sn},api_key.ECOFLOW_APP_KEY,api_key.ECOFLOW_SECRET_KEY),timeout=10)
    return r.json().get("data",{})

def eco_set(sn,params):
    payload={"sn":sn,"cmdId":17,"cmdFunc":254,"dirDest":1,"dirSrc":1,"dest":2,"needAck":True,"params":params}
    r=requests.put(f"{HOST}/iot-open/sign/device/quota",
        headers={**_sign(payload,api_key.ECOFLOW_APP_KEY,api_key.ECOFLOW_SECRET_KEY),
                 "Content-Type":"application/json"},
        json=payload,timeout=10)
    return r.json()

# ---- Tibber ----
def tibber_prices():
    q="{ viewer { homes { currentSubscription { priceInfo { today { total startsAt } tomorrow { total startsAt } }}}}}"
    r=requests.post("https://api.tibber.com/v1-beta/gql",
        json={"query":q},
        headers={"Authorization":f"Bearer {api_key.API_KEY}","Content-Type":"application/json"},
        timeout=15)
    for h in r.json()["data"]["viewer"]["homes"]:
        if h.get("currentSubscription"): return h["currentSubscription"]["priceInfo"]
    return {}

def price_map(pi):
    out={}
    for d in ("today","tomorrow"):
        for s in pi.get(d,[]) or []:
            ts=dt.datetime.fromisoformat(s["startsAt"]).replace(minute=0,second=0,microsecond=0)
            out[ts]=float(s["total"])
    return out

# ---- Load forecast (median by weekday+quarter) ----
def bucket(ts=None): return (int(ts or time.time())//900)*900

def log_load(ts,lw):
    c=sqlite3.connect(DB); c.execute("INSERT OR REPLACE INTO load_log(ts,load_w) VALUES(?,?)",(ts,lw)); c.commit(); c.close()

def median(xs):
    xs=sorted(xs); return xs[len(xs)//2] if xs else None

def forecast(slots,days=28):
    c=sqlite3.connect(DB)
    rows=c.execute("SELECT ts,load_w FROM load_log WHERE ts>=?",(bucket(time.time()-days*86400),)).fetchall()
    c.close()
    g={}
    for ts,lw in rows:
        t=dt.datetime.fromtimestamp(ts,tz=dt.timezone.utc).astimezone(TZ)
        k=(t.weekday(),(t.hour*60+t.minute)//15)
        g.setdefault(k,[]).append(lw)
    return [median(g.get((t.weekday(),(t.hour*60+t.minute)//15),[])) or 0 for t in slots]

# ---- Decision ----
def expensive(ph,slots):
    priced=[(s,ph.get(s.replace(minute=0))) for s in slots if ph.get(s.replace(minute=0))]
    if not priced: return set()
    ps=sorted(p for _,p in priced); thr=ps[int(len(ps)*(1-TOP_PERCENT))]
    return {s for s,p in priced if p>=thr}

def decide(soc,exp,slots,pred):
    if soc<=SOC_MIN: return SOC_MIN
    need=sum(l*0.25 for s,l in zip(slots,pred) if s in exp)
    avail=(soc-SOC_MIN)*BATT_CAPACITY_WH/100
    now=slots[0]
    return SOC_MIN if now in exp and avail>50 else soc+1

# ---- Main ----
def main():
    sn=eco_get_main_sn(api_key.ECOFLOW_DEVICE_ID)

    q=eco_quota(sn)
    soc=float(q.get("cmsBattSoc") or 0)
    log_load(bucket(),float(q.get("powGetSysLoad") or 0))

    eco_set(sn,{"cfgEnergyStrategyOperateMode":
        {"operateSelfPoweredOpen":True,"operateIntelligentScheduleModeOpen":False}})

    ph=price_map(tibber_prices())
    now=dt.datetime.now(TZ).replace(second=0,microsecond=0)
    slots=[now+dt.timedelta(minutes=15*i) for i in range(int(HORIZON_HOURS*4))]

    exp=expensive(ph,slots)
    pred=forecast(slots)

    target=clamp_reserve(decide(soc,exp,slots,pred))
    cur=q.get("backupReverseSoc")

    if cur is None or abs(target-int(cur))>=RESERVE_HYSTERESIS_PCT:
        eco_set(sn,{"cfgBackupReverseSoc":target})

    print(dt.datetime.now(), "SoC",soc,"→ Reserve",target)

if __name__=="__main__":
    main()
