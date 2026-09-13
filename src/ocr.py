"""EasyOCR wrapper."""
import easyocr


class OcrEngine:
    def __init__(self, languages=None, gpu=False):
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def read(self, frame, min_confidence: float = 0.4):
        """Run OCR on a BGR frame (as read by OpenCV).

        Returns a list of (text, confidence, box) tuples, where box is
        [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] in frame coordinates.
        """
        results = self.reader.readtext(frame)
        out = []
        for box, text, conf in results:
            if conf >= min_confidence:
                out.append((text, conf, box))
        return out
