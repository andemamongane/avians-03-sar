"""The live ground station: a person at a console while the episode runs.

    avians fly missions/sar_supply.yaml --ground live --realtime      # fast tier, slowed to real time
    avians fly missions/sar_supply.yaml --ground live --tier gazebo
    then open http://localhost:8765

The console (one HTML page, no dependencies) shows the drone's state, the
radio log and the pilot's pending questions, and lets the operator send a
message or tasking. Messages reach the drone through GroundStation.push()
with the modelled link latency, exactly like scripted ones, so an episode
flown with a live operator records the same way as any other.

Endpoints (JSON): GET /state, GET /log?since=N, POST /message {text, kind?,
to?}. Another program (a second model, a test) can drive the operator side
through them.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><meta charset="utf-8"><title>AVIANS ground station</title>
<style>body{font:14px system-ui;margin:16px;max-width:900px}#log{height:360px;overflow:auto;border:1px solid #ccc;
padding:8px;background:#fafafa;white-space:pre-wrap}.d{color:#1f5fd1}.g{color:#444}.o{color:#b3261e}
input,select,button{font:inherit;padding:6px}#text{width:60%}#state{font-family:ui-monospace,monospace;white-space:pre}</style>
<h2>Ground station</h2><div id=state>connecting...</div><h3>Radio log</h3><div id=log></div>
<p><select id=kind><option value=tasking>tasking</option><option value=answer>answer</option><option value=news>news</option></select>
<input id=text placeholder="message to the drone (Enter to send)"><button onclick=send()>Send</button></p>
<script>
let since=0;
async function poll(){try{const s=await (await fetch('/state')).json();
document.getElementById('state').textContent=s.text;
const l=await (await fetch('/log?since='+since)).json();
for(const m of l.messages){const d=document.createElement('div');d.className=m.sender=='ground'?'g':(m.sender=='operator'?'o':'d');
d.textContent=`t=${m.t.toFixed(0)}  ${m.sender} -> ${m.to}  [${m.kind}]  ${m.text}`;document.getElementById('log').appendChild(d);since=m.n+1;}
if(l.messages.length){const g=document.getElementById('log');g.scrollTop=g.scrollHeight;}}catch(e){}}
async function send(){const t=document.getElementById('text');if(!t.value)return;
await fetch('/message',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:t.value,kind:document.getElementById('kind').value})});t.value='';poll();}
document.getElementById('text').addEventListener('keydown',e=>{if(e.key=='Enter')send();});
setInterval(poll,1000);poll();
</script>"""


class Console:
    """The HTTP side of a GroundStation. `state_fn()` returns the text shown
    at the top (the simulator supplies it); `now_fn()` the sim time."""

    def __init__(self, station, state_fn, now_fn, port: int = 8765, host: str = "127.0.0.1"):
        self.station, self.state_fn, self.now_fn = station, state_fn, now_fn
        self.messages: list[dict] = []
        self.lock = threading.Lock()
        console = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    return self._json({"t": console.now_fn(), "text": console.state_fn(),
                                       "questions": [q for q in station.pending_questions if not q["answered"]]})
                if self.path.startswith("/log"):
                    since = int(self.path.split("since=")[1]) if "since=" in self.path else 0
                    with console.lock:
                        return self._json({"messages": [m for m in console.messages if m["n"] >= since]})
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                try:
                    d = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"error": "bad json"}, 400)
                text = str(d.get("text", "")).strip()
                if not text:
                    return self._json({"error": "empty"}, 400)
                kind = str(d.get("kind", "tasking"))
                station.push(console.now_fn(), "operator", kind, text, to=str(d.get("to", "all")))
                if kind == "answer":
                    for q in station.pending_questions:
                        q["answered"] = True
                return self._json({"ok": True})

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{host}:{port}"
        station.live = self

    def post(self, m) -> None:
        with self.lock:
            self.messages.append({"n": len(self.messages), "t": float(m.t), "sender": m.sender, "to": m.to,
                                  "kind": m.kind, "text": m.text, "wall": time.time()})

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
