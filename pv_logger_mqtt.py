#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, sqlite3, logging, datetime as dt, hmac, hashlib, ssl
import requests
from urllib.parse import urlencode

import paho.mqtt.client as mqtt
import api_key  # ECOFLOW_HOST, ECOFLOW_APP_KEY, ECOFLOW_SECRET_KEY, ECOFLOW_MIKRO_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DB_FILE = "/home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile/pv_data.db"

# --- welche Quotas wir wollen (gemäß EcoFlow Doku) ---
QUOTAS = [
    "20_1.pv1InputVolt",
    "20_1.pv1InputCur",
    "20_1.pv2InputVolt",
    "20_1.pv2InputCur",
]

# ---------- EcoFlow signing ----------
def _six_digit_nonce():
    return f"{int(time.time()*1000) % 900000 + 100000}"

def _flatten_params(obj, prefix=""):
    items = []
    if isinstance(obj, dict):
        for k in obj:
            key = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_params(obj[k], key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            items.extend(_flatten_params(v, key))
    else:
        items.append((prefix, "" if obj is None else str(obj)))
    return items

def _build_sign_string(params_dict, access_key, nonce, timestamp):
    kv = _flatten_params(params_dict) if params_dict else []
    kv.sort(key=lambda kv_: kv_[0])
    base = "&".join(f"{k}={v}" for k, v in kv) if kv else ""
    tail = f"accessKey={access_key}&nonce={nonce}&timestamp={timestamp}"
    return (base + "&" + tail) if base else tail

def _hmac_sha256_hex(secret_key, msg):
    return hmac.new(secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _signed_headers(access_key, secret_key, params_dict, content_type=None):
    ts = str(int(time.time()*1000))
    nonce = _six_digit_nonce()
    sign_str = _build_sign_string(params_dict, access_key, nonce, ts)
    sig = _hmac_sha256_hex(secret_key, sign_str)
    hdr = {"accessKey": access_key, "nonce": nonce, "timestamp": ts, "sign": sig}
    if content_type:
        hdr["Content-Type"] = content_type
    return hdr

def ecoflow_get_mqtt_cert():
    base = getattr(api_key, "ECOFLOW_HOST", "https://api-e.ecoflow.com").rstrip("/")
    path = "/iot-open/sign/certification"
    params = {}
    hdr = _signed_headers(api_key.ECOFLOW_APP_KEY, api_key.ECOFLOW_SECRET_KEY, params)
    url = f"{base}{path}"
    r = requests.get(url, headers=hdr, timeout=15)
    r.raise_for_status()
    j = r.json()
    if str(j.get("code")) != "0":
        raise RuntimeError(f"certification failed: {j}")
    data = j.get("data") or {}
    return {
        "account": data.get("certificateAccount"),
        "password": data.get("certificatePassword"),
        "host": data.get("url") or data.get("host"),
        "port": int(data.get("port") or 8883),
        "protocol": data.get("protocol") or "mqtts",
    }

# ---------- DB ----------
def db_init():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        # Wir nutzen deine vorhandene Tabelle pv_log (ts ist PRIMARY KEY)
        # Falls die Spalten pv1_w/pv2_w/pv_sum_w fehlen: dann ergänzt du sie einmalig per ALTER TABLE.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pv_log (
              ts INTEGER PRIMARY KEY,
              pv1_w REAL,
              pv2_w REAL,
              pv_sum_w REAL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pv_log_ts ON pv_log(ts);")
        conn.commit()
    finally:
        conn.close()

def insert_pv(ts_utc_epoch_s: int, pv1: float, pv2: float, pv_sum: float):
    # Minute-bucket, damit cron/loops nicht bei Sekunden-Duplikaten crashen
    ts_bucket = int(ts_utc_epoch_s // 60 * 60)

    conn = sqlite3.connect(DB_FILE, timeout=30)
    try:
        # PRIMARY KEY ts -> OR REPLACE verhindert UNIQUE constraint failed
        conn.execute(
            "INSERT OR REPLACE INTO pv_log(ts, pv1_w, pv2_w, pv_sum_w) VALUES (?,?,?,?)",
            (ts_bucket, float(pv1), float(pv2), float(pv_sum)),
        )
        conn.commit()
    finally:
        conn.close()

# ---------- PV parsing ----------
def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def pv_from_quota_payload(payload: dict):
    """
    Erwartet entweder:
    - payload["param"] mit Keys pv1InputVolt/pv1InputCur/... (Push)
    - oder payload["data"] mit Keys "20_1.pv1InputVolt" ... (get_reply je nach Implementierung)
    Wir unterstützen beides.
    """
    if not isinstance(payload, dict):
        return None

    # Case A: Push quota: {"cmdId":...,"cmdFunc":...,"param":{...}}
    param = payload.get("param")
    if isinstance(param, dict):
        v1 = _num(param.get("pv1InputVolt"))
        c1 = _num(param.get("pv1InputCur"))
        v2 = _num(param.get("pv2InputVolt"))
        c2 = _num(param.get("pv2InputCur"))
    else:
        # Case B: get_reply: {"data": {"20_1.pv1InputVolt":..., ...}}
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        v1 = _num(data.get("20_1.pv1InputVolt"))
        c1 = _num(data.get("20_1.pv1InputCur"))
        v2 = _num(data.get("20_1.pv2InputVolt"))
        c2 = _num(data.get("20_1.pv2InputCur"))

    def p(v, c):
        if v is None or c is None:
            return 0.0
        # Doku: 0.1 Einheiten => W = (V*I)/100
        return max(0.0, (v * c) / 100.0)

    pv1 = p(v1, c1)
    pv2 = p(v2, c2)
    return pv1, pv2, pv1 + pv2

# ---------- MQTT runtime ----------
class PVLoggerMQTT:
    def __init__(self, sn: str):
        self.sn = sn
        self.cert = None
        self.client = None

        self.online = None  # None/0/1
        self.last_pv = (0.0, 0.0, 0.0)
        self.last_seen_ts = 0

        self.topic_quota = None
        self.topic_status = None
        self.topic_get = None
        self.topic_get_reply = None

    def setup(self):
        self.cert = ecoflow_get_mqtt_cert()
        account = self.cert["account"]
        host = self.cert["host"]
        port = self.cert["port"]

        if not account or not self.cert["password"] or not host:
            raise RuntimeError(f"MQTT cert incomplete: {self.cert}")

        self.topic_quota = f"/open/{account}/{self.sn}/quota"
        self.topic_status = f"/open/{account}/{self.sn}/status"
        self.topic_get = f"/open/{account}/{self.sn}/get"
        self.topic_get_reply = f"/open/{account}/{self.sn}/get_reply"

        logging.info("MQTT cert ok: account=%s host=%s port=%s protocol=%s",
                     account, host, port, self.cert.get("protocol"))

        self.client = mqtt.Client(client_id=f"pv-logger-{self.sn}-{int(time.time())}", clean_session=True)
        self.client.username_pw_set(account, self.cert["password"])
        self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.tls_insecure_set(False)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self.client.connect(host, port, keepalive=60)

    def on_connect(self, client, userdata, flags, rc):
        logging.info("connected rc=%s", rc)
        client.subscribe(self.topic_quota, qos=1)
        client.subscribe(self.topic_status, qos=1)
        client.subscribe(self.topic_get_reply, qos=1)
        logging.info("subscribed %s", self.topic_quota)
        logging.info("subscribed %s", self.topic_status)
        logging.info("subscribed %s", self.topic_get_reply)

    def on_disconnect(self, client, userdata, rc):
        logging.warning("disconnected rc=%s", rc)

    def on_message(self, client, userdata, msg):
        try:
            raw = msg.payload.decode("utf-8", "ignore")
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        now = int(time.time())
        self.last_seen_ts = now

        if msg.topic.endswith("/status"):
            # Format: {"id":"..","version":"1.0","timestamp":...,"params":{"status":0/1}}
            st = None
            try:
                st = payload.get("params", {}).get("status")
            except Exception:
                st = None
            if st is not None:
                self.online = int(st)
                logging.info("status: online=%s", self.online)
            return

        if msg.topic.endswith("/quota") or msg.topic.endswith("/get_reply"):
            pv = pv_from_quota_payload(payload)
            if pv is None:
                return
            pv1, pv2, pv_sum = pv
            self.last_pv = (pv1, pv2, pv_sum)

            ts = int(time.time())
            insert_pv(ts, pv1, pv2, pv_sum)
            logging.info("PV logged (MQTT): pv1=%.1f W pv2=%.1f W sum=%.1f W", pv1, pv2, pv_sum)

    def request_quota(self):
        # Nur sinnvoll, wenn online (sonst kommt keine Reply)
        if self.online == 0:
            logging.info("skip get: device offline")
            return

        req = {
            "id": str(int(time.time()*1000)),
            "version": "1.0",
            "operateType": "TCP",
            "from": "python",
            "params": {"quotas": QUOTAS},
        }
        self.client.publish(self.topic_get, json.dumps(req), qos=1)
        logging.info("published get quotas to %s", self.topic_get)

    def run(self):
        # Netzwerkloop in Hintergrundthread
        self.client.loop_start()

        # zyklisch get request, falls keine quota pushs kommen
        while True:
            try:
                self.request_quota()
            except Exception as e:
                logging.error("request_quota failed: %s", e)

            # alle 60s einmal versuchen (bei online bekommst du dann Werte)
            time.sleep(60)

def main():
    db_init()
    sn = getattr(api_key, "ECOFLOW_MIKRO_ID", "").strip()
    if not sn:
        raise RuntimeError("ECOFLOW_MIKRO_ID fehlt in api_key.py")

    logger = PVLoggerMQTT(sn)
    logger.setup()
    logger.run()

if __name__ == "__main__":
    main()
