# LingBot Deploy WebUI

Static WebUI for the `lingbot_deploy_api.py` compatibility endpoint:

```text
/v1/lingbot/realtime
```

This UI speaks the current deploy protocol:

- client text JSON: `START`, `CONTROL`, `STOP`
- server text JSON: `STARTED`, `TIMING`, `FINISH`, `ERROR`
- server binary payload: raw `uint8` RGB24 frames

Open `index.html` directly in a browser, or serve this directory with any static
file server. The default WebSocket URL is `wss://127.0.0.1:8001/v1/lingbot/realtime`
when opened from `file://`.
