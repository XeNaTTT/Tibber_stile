# Tibber_stile
Tibber-Anzeige auf einem Waveshare 7.5 Inch E-Paper-Display.

## Tibber Pulse pro Witty-Pi-Lauf

Der Display-Lauf ermittelt das Realtime-Home über
`realTimeConsumptionEnabled`, liest genau einen `liveMeasurement`-Datensatz
und schließt den von Tibber gelieferten WebSocket anschließend. Snapshots und
die daraus berechneten Intervalle liegen in `tibber_snapshots.db`. Nur
Intervalle von 12 bis 18 Minuten, deren Endpunkte höchstens drei Minuten von
aufeinanderfolgenden Viertelstundengrenzen abweichen, werden im Chart genutzt.
Stündliche historische Werte bleiben als sparse Fallback-Punkte erhalten.

Eine bestimmte Home-ID kann optional (ohne sie einzuchecken) gesetzt werden:

```bash
export TIBBER_LIVE_HOME_ID='optional-home-id'
```

### Installation auf Raspberry Pi OS

Ein gemeinsames Projekt-venv vermeidet Änderungen am System-Python:

```bash
cd /home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libopenblas-dev libjpeg-dev
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Die vorhandenen `run_tibber.sh`/`update_and_run_tibber.sh` sollten danach
denselben einzelnen Display-Lauf mit dem venv-Interpreter starten (kein neuer
Daemon und kein Logger):

```bash
cd /home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile
exec .venv/bin/python Tibber_stile.py
```

Die beiden Boot-Skripte waren in der Entwicklungsumgebung nicht vorhanden und
wurden deshalb nicht automatisch verändert.

### Manueller Live-Test und Diagnose

Der reguläre einmalige Live-Test ist ein normaler Display-Lauf:

```bash
cd /home/alex/E-Paper-tibber-Preisanzeige/Tibber_stile
.venv/bin/python Tibber_stile.py
```

Danach lässt sich der persistierte Stand ohne Netzwerkzugriff anzeigen:

```bash
.venv/bin/python tibber_live.py --status
```

Beim ersten Wakeup wird nur ein Snapshot gespeichert. Beim zweiten entsteht
ein Intervall: Liegen Abstand und Viertelstundengrenzen in den Toleranzen,
wird es als `valid_quarter` gezeichnet; andernfalls bleibt es zur Diagnose als
`gap`, `counter_reset` oder `invalid` gespeichert und der HOURLY-Punkt bleibt
der Chart-Fallback.

## Tests
Run a lightweight compile check to ensure the Python sources are syntactically valid:

```bash
./run_tests.sh
```
