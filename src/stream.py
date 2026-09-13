"""RTSP stream handling for the Wyze cam.

Uses a background thread to continuously grab frames, so callers (the
display loop, OCR loop) always get the most recent frame instantly
instead of blocking on network/decode latency every time they ask for
one. Without this, a laggy/jittery network connection makes the whole
pipeline - including the --show preview - feel choppy even though OCR
itself is fast.
"""
import threading
import time
import cv2


class RtspStream:
    def __init__(self, url: str, reconnect_delay: float = 3.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stopped = False
        self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        # Timeouts must be passed as constructor params, not set via
        # .set() afterward - the constructor itself blocks trying to
        # establish the RTSP connection, and .set() only takes effect
        # once it's already open, which is too late if it's stuck.
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000,
        ]
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        # Keep internal buffer small so we get recent frames, not stale ones.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _loop(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._ok = True
            else:
                with self._lock:
                    self._ok = False
                print(f"[stream] connection lost, retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self._open()

    def read(self):
        """Return (ok, frame) - the most recently grabbed frame, instantly."""
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def release(self):
        # Stop the background thread and wait for it to actually exit
        # before touching self.cap from this thread - releasing cap while
        # the background thread is mid-read() on the same object is a
        # race that can corrupt the RTSP teardown, leaving a session
        # stuck open on the *camera's* side (which then refuses new
        # connections on the next run, even though our process exited).
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if self.cap:
            self.cap.release()
