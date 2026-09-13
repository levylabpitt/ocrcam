"""Main loop: pull frames from the Wyze RTSP stream, run OCR periodically,
log results, and optionally show a live annotated preview window.
"""
import argparse
import csv
import difflib
import os
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv

from .stream import RtspStream
from .ocr import OcrEngine
from .roi import load_roi
from .preprocess import enhance, is_too_blurry
from .stabilize import TemporalStabilizer

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_CSV = os.path.join(ROOT, "logs", "ocr_results.csv")


def safe_print(msg: str):
    """Print without crashing on characters the console encoding can't
    display (e.g. a stray OCR misread of a non-Latin glyph on Windows'
    default cp1252 console)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def crop(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def draw_boxes(frame, roi, results):
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    for text, conf, box in results:
        pts = [(int(px) + ox, int(py) + oy) for px, py in box]
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cv2.putText(frame, text, (pts[0][0], pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def ensure_log():
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "text", "confidence"])


def log_results(results):
    if not results:
        return
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        ts = datetime.now().isoformat(timespec="seconds")
        for text, conf, _box in results:
            writer.writerow([ts, text, f"{conf:.3f}"])


def run(interval: float, show: bool, min_confidence: float, save_frames: bool,
        blur_threshold: float, stabilize_window: int, stabilize_hits: int,
        log_cooldown: float):
    load_dotenv()
    url = os.environ["RTSP_URL"]
    roi = load_roi()
    if roi:
        print(f"[main] using saved ROI: {roi}")
    else:
        print("[main] no ROI saved, OCR-ing full frame "
              "(run `python -m src.roi` to select a region)")

    # Load OCR models before opening the stream - model loading can take
    # 20-30s and an idle, unread RTSP connection times out during that wait.
    ocr = OcrEngine()
    stream = RtspStream(url)
    stabilizer = TemporalStabilizer(window=stabilize_window, min_hits=stabilize_hits)
    recently_logged = []  # list of (text, timestamp) - fuzzy-deduped, not exact
    ensure_log()

    last_ocr = 0.0
    last_heartbeat = 0.0
    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                # non-blocking now (threaded stream) - avoid a hot spin
                # while waiting for the first/next frame.
                time.sleep(0.05)
                continue

            now = time.time()
            results = []
            if now - last_ocr >= interval:
                last_ocr = now
                region = crop(frame, roi)

                if is_too_blurry(region, blur_threshold):
                    print("[main] frame too blurry, skipping OCR pass")
                else:
                    processed = enhance(region)
                    raw_results = ocr.read(processed, min_confidence=min_confidence)
                    # scale boxes back down since `enhance` may have upscaled
                    scale = processed.shape[1] / region.shape[1]
                    results = [
                        (t, c, [(px / scale, py / scale) for px, py in box])
                        for t, c, box in raw_results
                    ]

                    for text, conf, _ in results:
                        safe_print(f"[ocr] '{text}' ({conf:.2f})")

                    # Confirmed = a cluster of similar-looking reads (not
                    # necessarily an exact string) hit min_hits within the
                    # window. Dedup against recently-logged text (fuzzy, not
                    # exact - the representative can drift between passes)
                    # so a persistent sign doesn't spam the CSV every pass.
                    confirmed = stabilizer.update([t for t, _, _ in results])
                    recently_logged[:] = [
                        (t, ts) for t, ts in recently_logged if now - ts < log_cooldown
                    ]
                    to_log = []
                    for rep in confirmed:
                        is_dupe = any(
                            difflib.SequenceMatcher(None, rep.lower(), t.lower()).ratio() >= 0.6
                            for t, _ in recently_logged
                        )
                        if not is_dupe:
                            recently_logged.append((rep, now))
                            to_log.append((rep, 1.0, None))
                    if to_log:
                        for rep, _, _ in to_log:
                            safe_print(f"[confirmed] '{rep}'")
                        log_results(to_log)
                        if save_frames:
                            fname = os.path.join(
                                ROOT, "logs",
                                f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                            )
                            cv2.imwrite(fname, frame)

                # Heartbeat so "no text found" doesn't look identical to
                # "stuck" - print a status line periodically when nothing's
                # being detected.
                if not results and now - last_heartbeat >= 5.0:
                    last_heartbeat = now
                    print("[main] watching... no text detected in ROI right now")

            if show:
                annotated = draw_boxes(frame.copy(), roi, results) if results else frame
                if roi:
                    x, y, w, h = roi
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("wyzeocr", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # cv2.waitKey() above already paces the --show loop;
                # without it, cap the loop so it doesn't spin at 100% CPU
                # re-reading/re-checking the same frame.
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stream.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Wyze cam RTSP -> OCR")
    parser.add_argument("--interval", type=float, default=0.5,
                         help="Seconds between OCR passes (default: 0.5)")
    parser.add_argument("--show", action="store_true",
                         help="Show a live annotated preview window")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                         help="Minimum OCR confidence to keep a result (default: 0.4)")
    parser.add_argument("--save-frames", action="store_true",
                         help="Save a jpg to logs/ each time OCR finds text")
    parser.add_argument("--blur-threshold", type=float, default=60.0,
                         help="Skip OCR on frames blurrier than this (default: 60.0, lower = more permissive)")
    parser.add_argument("--stabilize-window", type=int, default=3,
                         help="How many recent OCR passes to consider for voting (default: 3)")
    parser.add_argument("--stabilize-hits", type=int, default=2,
                         help="How many of those passes must agree before logging (default: 2)")
    parser.add_argument("--log-cooldown", type=float, default=20.0,
                         help="Seconds before re-logging the same confirmed text (default: 20.0)")
    args = parser.parse_args()
    run(args.interval, args.show, args.min_confidence, args.save_frames,
        args.blur_threshold, args.stabilize_window, args.stabilize_hits,
        args.log_cooldown)


if __name__ == "__main__":
    main()
