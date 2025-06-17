
"""
Solo la lógica de análisis visual:
- Colores dominantes
- Formas geométricas (círculo, rectángulo, triángulo, cuadrilátero)
"""

import cv2
import numpy as np


class ImageAnalysisService:
    # ───────────────────────────
    # Colores dominantes
    # ───────────────────────────
    def detect_dominant_colors(self, img_bgr, k: int = 5):
        small = cv2.resize(img_bgr, (200, 200))
        pixels = small.reshape((-1, 3)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten())

        return [
            {
                "rgb": centers[i].tolist(),
                "percent": int(100 * cnt / sum(counts)),
            }
            for i, cnt in enumerate(counts)
        ]

    # ───────────────────────────
    # Formas geométricas simples
    # ───────────────────────────
    def detect_shapes(self, img_bgr, min_area: int = 150):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                stype = "triangle"
            elif len(approx) == 4:
                ar = w / float(h)
                stype = "rectangle" if 0.8 < ar < 1.2 else "quadrilateral"
            elif len(approx) > 8:
                stype = "circle"
            else:
                stype = "unknown"

            shapes.append(
                {"type": stype, "bbox": [x, y, x + w, y + h], "area": int(area)}
            )
        return shapes
