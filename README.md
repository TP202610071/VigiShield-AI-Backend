# VigiShield AI Backend

Processes the IP camera feed and sends detected security events to the ASP.NET Core backend.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            Local Machine (Home)           │
                    │                                          │
  IP Camera ──RTSP──► Python AI Backend                       │
      │               (this service)                          │
      │                    │                                   │
      │              process frames                            │
      │              send events ──HTTP──► ASP.NET Backend     │
      │                                                        │
      │  Option B only:                                        │
      │  FFmpeg ──RTMP──► Server MediaMTX ──HLS──► Flutter App │
      └──────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in config
cp .env.example .env
# Edit .env: set BACKEND_API_URL, INTERNAL_API_KEY, HOUSEHOLD_ID

# 3. Run
python main.py
```

## Stream Modes

### Option A — Direct RTSP (fixed IP / local network)
The Python backend connects directly to the camera's RTSP stream.
Configure the camera in the VigiShield app first, then the backend
will fetch the RTSP URL automatically.

```
HOUSEHOLD_ID=your-uuid
BACKEND_API_URL=http://your-server:5020
INTERNAL_API_KEY=your-key
```

### Option B — RTMP Relay (CGNAT / dynamic IP)
The camera is on a local network without a fixed public IP.
Python still reads the RTSP from the camera directly.
A separate FFmpeg process relays the stream to the server for Flutter to watch.

**FFmpeg relay command** (get exact URL from the app → Settings → Cámara):
```bash
ffmpeg -i rtsp://user:pass@camera-ip:554/stream \
       -c copy -f flv \
       rtmp://your-server:1935/live/your-stream-key
```

### Manual RTSP override (skip backend config)
```
RTSP_URL=rtsp://admin:password@192.168.1.100:554/h264/ch1/main/av_stream
```

## Current State: Simulation Mode

Real AI models are not yet integrated. The `EventDetector` emits random events
every `EVENT_SIMULATION_INTERVAL_SECONDS` seconds to test the pipeline.

To integrate real models, edit `event_detector.py → EventDetector.process_frame()`.
See the docstring there for detailed implementation guidance.

## Common Camera RTSP Paths

| Brand       | Path                                     |
|-------------|------------------------------------------|
| Hikvision   | `/h264/ch1/main/av_stream`               |
| Dahua       | `/cam/realmonitor?channel=1&subtype=0`   |
| Reolink     | `/h264Preview_01_main`                   |
| Amcrest     | `/cam/realmonitor?channel=1&subtype=0`   |
| Generic     | `/stream`, `/video`, `/live`             |

## MediaMTX Setup (for Option B server side)

Install MediaMTX on your server:
```bash
# Download from https://github.com/bluenviron/mediamtx/releases
./mediamtx  # runs with default config on ports 8888 (HLS) and 1935 (RTMP)
```

Then set in backend `appsettings.json`:
```json
"MediaMtx": {
  "HlsBaseUrl": "http://your-server-ip:8888",
  "ServerHost": "your-server-ip",
  "RtmpPort": "1935"
}
```
