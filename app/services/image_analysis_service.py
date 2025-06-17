"""
Servicio para analizar planos eléctricos
----------------------------------------
- Detección de símbolos (modelo YOLOv5 .pt)
- Detección de rutas (contornos / líneas de cableado)
- Detección de textos (OCR con Tesseract)
- Detección de colores dominantes
- Detección de formas geométricas básicas
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
import pytesseract


class ImageAnalysisService:

    #  INIT → carga el modelo YOLOv5

    def __init__(self, yolo_weights: str = "weights/electrical_best.pt"):
        """
        yolo_weights: ruta a un modelo YOLOv5 entrenado con símbolos eléctricos.
        """
        w_path = Path(yolo_weights)
        if not w_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo YOLOv5 en '{yolo_weights}'."
            )

        # Carga modelo YOLOv5 usando torch.hub (fuente github)
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=str(w_path), source="github"
        )
        self.model.conf = 0.3  # confianza mínima


    # 1) DETECCIÓN DE SÍMBOLOS (YOLOv5)

    def detect_symbols(self, img_bgr):
        # YOLOv5 espera RGB; convertimos antes de predecir
        results = self.model(img_bgr[:, :, ::-1])
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        symbols = []
        for det in detections:
            symbols.append(
                {
                    "type": det["name"],
                    "bbox": [
                        int(det["xmin"]),
                        int(det["ymin"]),
                        int(det["xmax"]),
                        int(det["ymax"]),
                    ],
                    "conf": float(det["confidence"]),
                }
            )
        return symbols


    # 2) DETECCIÓN DE RUTAS (cableado)
    
    def detect_routes(self, img_bgr, min_length=200):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        routes = []
        for cnt in contours:
            length = cv2.arcLength(cnt, False)
            if length > min_length:
                routes.append(cnt.squeeze().tolist())
        return routes

    
    # 3) OCR DE TEXTOS  (con filtro de ruido)
    
    def detect_texts(self, img_bgr):
        pil_img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ocr = pytesseract.image_to_data(
            pil_img, lang="spa", config="--psm 6", output_type=pytesseract.Output.DICT
        )
        texts = []
        for i in range(len(ocr["text"])):
            if int(ocr["conf"][i]) < 60:
                continue
            word = ocr["text"][i].strip()
            if len(word) < 2:
                continue
            # descarta palabras con demasiados símbolos (ruido)
            if sum(c.isalnum() for c in word) / len(word) < 0.6:
                continue

            x, y, w, h = (
                ocr["left"][i],
                ocr["top"][i],
                ocr["width"][i],
                ocr["height"][i],
            )
            texts.append({"text": word, "bbox": [x, y, x + w, y + h]})
        return texts

    
    # 4) COLORES DOMINANTES (k‑means)
    
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
            {"rgb": centers[i].tolist(), "percent": int(100 * c / counts.sum())}
            for i, c in enumerate(counts)
        ]

    
    # 5) FORMAS GEOMÉTRICAS BÁSICAS
    
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

    
    # 6) DESCRIPCIÓN NATURAL DEL ANÁLISIS
    
    @staticmethod
    def generar_descripcion(analysis: dict) -> str:
        texts = [t["text"] for t in analysis.get("texts", [])]
        colors = analysis.get("colors", [])
        routes = analysis.get("routes", [])

        partes = ["La imagen corresponde a un plano eléctrico."]

        # Palabras clave legibles (máx 5)
        palabras = [t for t in texts if len(t) > 2][:5]
        if palabras:
            partes.append(f"Se identificaron palabras como: {', '.join(palabras)}.")

        # Cantidad de rutas
        if routes:
            partes.append(f"Se detectaron aproximadamente {len(routes)} rutas de cableado.")

        # Color predominante
        if colors:
            principal = colors[0]['rgb']
            partes.append(f"El color predominante es RGB {principal}.")

        return " ".join(partes)

    
    # MÉTODO PRINCIPAL
    
    def analyze_image(self, image_path: str) -> dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"error": "No se pudo leer la imagen."}

        analysis = {
            "symbols": self.detect_symbols(img_bgr),
            "routes": self.detect_routes(img_bgr),
            "texts": self.detect_texts(img_bgr),
            "colors": self.detect_dominant_colors(img_bgr),
            "shapes": self.detect_shapes(img_bgr),
        }

        summary = self.generar_descripcion(analysis)
        return {"raw": analysis, "summary": summary}
