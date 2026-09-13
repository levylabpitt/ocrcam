"""Temporal stabilization: OCR on a single frame is noisy (motion blur,
compression, lighting flicker), and on top of that, a garbled read of the
same physical text often comes out as a *different* string every pass
(e.g. "Weile", "Wettle", "Peatife" for the same sign). Exact-match voting
would never confirm anything in that case, so this clusters similar-looking
strings together and confirms the cluster, not the exact string.
"""
import difflib
from collections import deque, Counter


class TemporalStabilizer:
    def __init__(self, window: int = 3, min_hits: int = 2, similarity: float = 0.6):
        """
        window: how many recent OCR passes to remember
        min_hits: how many hits (in that window) a cluster needs before
                  it's "confirmed" and gets logged
        similarity: 0-1 fuzzy match threshold for treating two OCR reads
                    as "the same text" (difflib ratio)
        """
        self.window = window
        self.min_hits = min_hits
        self.similarity = similarity
        self.history = deque(maxlen=window)

    def _cluster(self, all_texts):
        clusters = []
        for t in all_texts:
            best = None
            best_ratio = 0.0
            for c in clusters:
                ratio = difflib.SequenceMatcher(None, t.lower(), c[0].lower()).ratio()
                if ratio > best_ratio:
                    best, best_ratio = c, ratio
            if best is not None and best_ratio >= self.similarity:
                best.append(t)
            else:
                clusters.append([t])
        return clusters

    def update(self, texts):
        """texts: list of strings read in the latest OCR pass.
        Returns a list of "confirmed" representative strings - one per
        cluster of similar reads that has hit `min_hits` within the window.
        The representative is the most frequent exact string in that
        cluster (ties broken by length, since garbled reads are often
        truncated).
        """
        self.history.append(list(texts))
        all_texts = [t for frame in self.history for t in frame]
        clusters = self._cluster(all_texts)

        confirmed = []
        for c in clusters:
            if len(c) >= self.min_hits:
                counts = Counter(c)
                top_count = max(counts.values())
                candidates = [t for t, n in counts.items() if n == top_count]
                rep = max(candidates, key=len)
                confirmed.append(rep)
        return confirmed
