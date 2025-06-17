import torch
import cv2
from pathlib import Path

class YoloV5Detector:
    def __init__(self, weights_path='weights/electrical_best.pt'):
        self.model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')
        self.model.conf = 0.4  # confianza mínima

    def detect(self, image_path):
        results = self.model(image_path)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        parsed = []
        for det in detections:
            parsed.append({
                "type": det["name"],
                "conf": float(det["confidence"]),
                "bbox": [int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])]
            })
        return parsed
