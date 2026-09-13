# wyzeocr

Pulls the RTSP feed from a Wyze v3 camera and runs OCR (PaddleOCR,
PP-OCRv4 mobile models) on it periodically, logging confirmed text to
`logs/ocr_results.csv`.

## Setup

Already done for you:
- `venv/` created, all dependencies installed
- `.env` contains your `RTSP_URL`

If you ever need to reinstall:
```powershell
cd wyzeocr
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

**1. (Optional) Select a region of interest.** If you only want to OCR
part of the frame (a meter, sign, package label, etc.) instead of the
whole image:

```powershell
python -m src.roi
```

A window opens with the live frame — drag a box, hit ENTER. Saved to
`.roi.json` and used automatically afterward. Skip this to OCR the
full frame (slower, noisier).

**2. Run the pipeline:**

```powershell
python -m src.main --show
```

Flags:
- `--show` — live preview window with OCR boxes drawn on it (press `q` to quit)
- `--interval 2.0` — seconds between OCR passes
- `--min-confidence 0.4` — drop low-confidence individual reads
- `--stabilize-window 3` / `--stabilize-hits 2` — a text cluster needs
  this many similar-looking hits within this many recent passes before
  it's "confirmed" and logged (OCR on a live feed is noisy - the same
  sign often reads as a slightly different garbled string each pass)
- `--log-cooldown 20.0` — seconds before re-logging the same confirmed
  text, so a persistent sign doesn't spam the CSV every pass
- `--blur-threshold 60.0` — skip OCR on frames blurrier than this
- `--save-frames` — save a jpg to `logs/` each time text is confirmed

Results append to `logs/ocr_results.csv` (timestamp, text, confidence).

## Known limitations

- **CPU-only speed**: a single OCR pass takes ~5-10s on this machine.
  PaddlePaddle's CPU accelerator (MKL-DNN) crashes on this hardware, so
  it's disabled (`enable_mkldnn=False` in `src/ocr.py`) - without it,
  the more accurate "medium" models take *minutes* per frame, which is
  why this uses the smaller/faster "mobile" models instead. If you get
  a GPU, switch `OcrEngine(accurate=True)` and swap mkldnn for
  `device="gpu"` for much better accuracy.
- **Accuracy**: the mobile models still misread things sometimes
  (`DELL` → `DULL`). The temporal stabilizer helps filter out one-off
  noise but won't fix a consistently-misread font/angle.

## Troubleshooting

**`python -m src.main` hangs at startup, right after the model-loading
messages, before any `[ocr]` lines print.**

This almost always means a Wyze camera's RTSP connection is stuck open
from a previous run - Wyze cams typically only allow **one RTSP viewer
at a time**, and if a previous `python` process was killed uncleanly
(closed terminal, force-killed, crashed mid-run) it can leave the
connection held open, so your new run blocks forever waiting for the
camera to accept a connection it's not going to give up.

Fix: close any other terminals running this script, then check for and
kill leftover processes:
```powershell
Get-Process python | Stop-Process -Force
```
Then try again. (`stream.py` now sets connection timeouts, so future
hangs should fail and retry after ~8s instead of hanging forever - but
a genuinely stuck camera-side connection from an old process still
needs to be killed manually.)

- `RtspStream` auto-reconnects if the Wyze stream drops mid-run.
- Credentials live in `.env` (gitignored) - never hardcoded in source.
- First run downloads PaddleOCR's model weights (a few hundred MB), so
  it's slower to start once.
