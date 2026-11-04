"""Scene-cut (camera switch / reconnect) detector for ATCC.

CCTV walls, compilations and reconnecting streams jump to a different camera, often with a
~1 s dissolve. Tracker IDs, vehicle trajectories and accident incidents from the previous scene
must not leak into the new one. The current frame is compared with the frame ~1 s earlier; a
cut is declared when colour histogram, edge layout AND pixel content all changed strongly
(a flash of emergency lights changes colour but not the edge layout, so it is ignored).
"""
from collections import deque

import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, fps: float = 30.0, stride: int = 1,
                 hist_corr_thresh: float = 0.5, edge_corr_thresh: float = 0.35,
                 pixel_diff_thresh: float = 30.0, min_gap_seconds: float = 3.0):
        self.lookback = max(3, int(round(fps / max(1, stride))))      # ~1 second of processed frames
        self.hist_corr_thresh = hist_corr_thresh
        self.edge_corr_thresh = edge_corr_thresh
        self.pixel_diff_thresh = pixel_diff_thresh
        self.min_gap = max(self.lookback * 3, int(min_gap_seconds * fps / max(1, stride)))
        self._buf = deque(maxlen=self.lookback + 1)
        self._since_cut = 10 ** 9

    def reset(self):
        self._buf.clear()
        self._since_cut = 10 ** 9

    @staticmethod
    def _features(frame):
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        e = np.sqrt(gx * gx + gy * gy).ravel()
        e = e - e.mean()
        e /= (np.linalg.norm(e) + 1e-6)
        return hist, e, gray

    def update(self, frame) -> bool:
        """Feed every processed frame. Returns True when this frame belongs to a NEW scene."""
        feat = self._features(frame)
        self._since_cut += 1
        cut = False
        if len(self._buf) == self._buf.maxlen:
            h0, e0, g0 = self._buf[0]
            h1, e1, g1 = feat
            hist_corr = cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL)
            edge_corr = float(np.dot(e0, e1))
            diff = float(np.mean(np.abs(g0 - g1)))
            if (hist_corr < self.hist_corr_thresh and edge_corr < self.edge_corr_thresh
                    and diff > self.pixel_diff_thresh and self._since_cut >= self.min_gap):
                cut = True
                self._since_cut = 0
                self._buf.clear()
        self._buf.append(feat)
        return cut
