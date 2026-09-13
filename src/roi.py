"""Interactive ROI (region of interest) selection, saved to .roi.json.

Run this once to tell the pipeline which part of the frame to run OCR
on (e.g. a meter display, a sign, a package label). If no ROI is saved,
main.py falls back to OCR-ing the full frame.
"""
import json
import os
import time
import cv2

from .stream import RtspStream

ROI_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".roi.json")


def select_roi(url: str, wait_timeout: float = 10.0):
    stream = RtspStream(url)
    print("[roi] connecting and grabbing a frame...")
    # RtspStream grabs frames on a background thread - the very first
    # frame isn't available the instant the object is constructed, so
    # poll briefly instead of reading once and giving up.
    deadline = time.time() + wait_timeout
    ok, frame = stream.read()
    while not ok and time.time() < deadline:
        time.sleep(0.2)
        ok, frame = stream.read()
    if not ok:
        stream.release()
        raise RuntimeError(
            f"Could not grab a frame from the stream within {wait_timeout}s."
        )

    print("[roi] got a frame - drag a box, then press ENTER to confirm.")

    box = cv2.selectROI("Select OCR region - ENTER to confirm, ESC to cancel", frame)
    cv2.destroyAllWindows()
    stream.release()

    x, y, w, h = [int(v) for v in box]
    if w == 0 or h == 0:
        print("[roi] no region selected, nothing saved.")
        return None

    with open(ROI_PATH, "w") as f:
        json.dump({"x": x, "y": y, "w": w, "h": h}, f)
    print(f"[roi] saved region {x, y, w, h} to {ROI_PATH}")
    return x, y, w, h


def load_roi():
    if not os.path.exists(ROI_PATH):
        return None
    with open(ROI_PATH) as f:
        d = json.load(f)
    return d["x"], d["y"], d["w"], d["h"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    select_roi(os.environ["RTSP_URL"])
