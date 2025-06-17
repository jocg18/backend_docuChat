"""Servicio avanzado de análisis de imágenes para planos eléctricos

Este servicio proporciona:
- Detección avanzada de símbolos eléctricos
- Análisis inteligente de rutas y conexiones
- Extracción mejorada de texto con contexto
- Detección específica de componentes eléctricos
- Generación de descripciones detalladas en lenguaje natural
- Análisis de colores y formas especializados
"""

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image as PILImage
import re
from typing import Dict, List, Tuple, Optional
import math

class AdvancedImageAnalysisService:
    def __init__(self):
        """Inicializa el servicio con modelos y configuraciones optimizadas"""
        # Configuración de Tesseract para mejor OCR
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=.,:;()[]{}"\'/\\'
        
        # Diccionario de símbolos eléctricos comunes
        self.electrical_symbols = {
            'interruptor': ['switch', 'SW', 'S'],
            'tomacorriente': ['outlet', 'receptacle', 'R'],
            'luminaria': ['light', 'lamp', 'L'],
            'panel_electrico': ['panel', 'board', 'P'],
            'motor': ['motor', 'M'],
            'transformador': ['transformer', 'T'],
            'fusible': ['fuse', 'F'],
            'breaker': ['breaker', 'CB'],
            'medidor': ['meter', 'kWh'],
            'tierra': ['ground', 'GND', 'earth']
        }
        
        # Cargar modelo YOLOv5 optimizado
        try:
            w_path = "./weights/yolov5s.pt"  # Usar modelo estándar si no hay uno específico
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=w_path, source="github", trust_repo=True
            )
            self.model.conf = 0.25  # Confianza mínima
            self.model.iou = 0.45   # IoU threshold
        except Exception as e:
            print(f"⚠️ Error cargando YOLOv5: {e}")
            self.model = None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DETECCIÓN AVANZADA DE SÍMBOLOS ELÉCTRICOS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_advanced_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección avanzada de símbolos eléctricos combinando YOLOv5 y análisis morfológico"""
        symbols = []
        
        # 1. Detección con YOLOv5 si está disponible
        if self.model is not None:
            yolo_symbols = self._detect_yolo_symbols(img_bgr)
            symbols.extend(yolo_symbols)
        
        # 2. Detección de formas específicas eléctricas
        geometric_symbols = self._detect_electrical_shapes(img_bgr)
        symbols.extend(geometric_symbols)
        
        # 3. Detección por patrones de texto
        text_symbols = self._detect_text_based_symbols(img_bgr)
        symbols.extend(text_symbols)
        
        # 4. Filtrar duplicados y mejorar clasificación
        symbols = self._filter_and_classify_symbols(symbols)
        
        return symbols
    
    def _detect_yolo_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección usando YOLOv5 con mapeo a símbolos eléctricos"""
        symbols = []
        try:
            results = self.model(img_bgr)
            
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > 0.25:
                    yolo_class = self.model.names[int(cls)]
                    electrical_type = self._map_yolo_to_electrical(yolo_class)
                    
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    
                    symbols.append({
                        "type": electrical_type,
                        "subtype": yolo_class,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "area": area,
                        "detection_method": "yolo"
                    })
        except Exception as e:
            print(f"⚠️ Error en detección YOLO: {e}")
        
        return symbols
    
    def _detect_electrical_shapes(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección específica de formas geométricas típicas en planos eléctricos"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de círculos (conexiones, luminarias)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                symbols.append({
                    "type": "conexion_circular",
                    "subtype": "junction_point",
                    "confidence": 0.8,
                    "bbox": [x-r, y-r, x+r, y+r],
                    "area": int(math.pi * r * r),
                    "detection_method": "geometric"
                })
        
        # Detección de rectángulos (paneles, componentes)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                
                if len(approx) == 4 and 0.7 < w/h < 1.3:
                    symbols.append({
                        "type": "componente_rectangular",
                        "subtype": "electrical_component",
                        "confidence": 0.7,
                        "bbox": [x, y, x+w, y+h],
                        "area": int(area),
                        "detection_method": "geometric"
                    })
        
        return symbols
    
    def _detect_text_based_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección de símbolos basada en texto identificado"""
        symbols = []
        
        # Extraer texto de la imagen
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        text_data = pytesseract.image_to_data(gray, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
        
        for i, text in enumerate(text_data['text']):
            if text.strip():
                conf = int(text_data['conf'][i])
                if conf > 30:
                    # Buscar coincidencias con símbolos eléctricos
                    electrical_type = self._classify_text_as_electrical(text)
                    
                    if electrical_type:
                        x, y, w, h = (
                            text_data['left'][i], text_data['top'][i],
                            text_data['width'][i], text_data['height'][i]
                        )
                        
                        symbols.append({
                            "type": electrical_type,
                            "subtype": "text_identified",
                            "confidence": conf / 100.0,
                            "bbox": [x, y, x+w, y+h],
                            "area": w * h,
                            "text": text,
                            "detection_method": "text"
                        })
        
        return symbols
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS AVANZADO DE RUTAS Y CONEXIONES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_advanced_routes(self, img_bgr: np.ndarray) -> List[Dict]:
        """Análisis avanzado de rutas eléctricas y conexiones"""
        routes = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Detección de líneas usando HoughLines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            # Clasificar líneas por orientación y longitud
            horizontal_lines = []
            vertical_lines = []
            diagonal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                
                route_data = {
                    "start": [x1, y1],
                    "end": [x2, y2],
                    "length": length,
                    "angle": angle,
                    "type": ""
                }
                
                if -15 <= angle <= 15 or 165 <= abs(angle) <= 180:
                    route_data["type"] = "horizontal_cable"
                    horizontal_lines.append(route_data)
                elif 75 <= abs(angle) <= 105:
                    route_data["type"] = "vertical_cable"
                    vertical_lines.append(route_data)
                else:
                    route_data["type"] = "diagonal_cable"
                    diagonal_lines.append(route_data)
            
            routes.extend(horizontal_lines[:10])  # Limitar para evitar ruido
            routes.extend(vertical_lines[:10])
            routes.extend(diagonal_lines[:5])
        
        # 2. Detección de intersecciones
        intersections = self._detect_intersections(routes)
        
        return {
            "cables": routes,
            "intersections": intersections,
            "summary": {
                "total_cables": len(routes),
                "horizontal": len([r for r in routes if r["type"] == "horizontal_cable"]),
                "vertical": len([r for r in routes if r["type"] == "vertical_cable"]),
                "intersections": len(intersections)
            }
        }
    
    def _detect_intersections(self, routes: List[Dict]) -> List[Dict]:
        """Detecta intersecciones entre rutas"""
        intersections = []
        
        for i, route1 in enumerate(routes):
            for j, route2 in enumerate(routes[i+1:], i+1):
                intersection = self._line_intersection(route1, route2)
                if intersection:
                    intersections.append({
                        "point": intersection,
                        "routes": [i, j],
                        "type": "cable_intersection"
                    })
        
        return intersections
    
    def _line_intersection(self, route1: Dict, route2: Dict) -> Optional[List[int]]:
        """Calcula intersección entre dos líneas"""
        x1, y1 = route1["start"]
        x2, y2 = route1["end"]
        x3, y3 = route2["start"]
        x4, y4 = route2["end"]
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 0.001:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return [int(x), int(y)]
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXTRACCIÓN INTELIGENTE DE TEXTO
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def extract_intelligent_text(self, img_bgr: np.ndarray) -> Dict:
        """Extracción inteligente de texto con contexto eléctrico"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Mejorar imagen para OCR
        processed = self._preprocess_for_ocr(gray)
        
        # Extraer texto con información de posición
        text_data = pytesseract.image_to_data(
            processed, config=self.tesseract_config, output_type=pytesseract.Output.DICT
        )
        
        # Procesar y clasificar texto
        text_elements = []
        electrical_labels = []
        dimensions = []
        notes = []
        
        for i, text in enumerate(text_data['text']):
            if text.strip() and int(text_data['conf'][i]) > 30:
                element = {
                    "text": text.strip(),
                    "confidence": int(text_data['conf'][i]),
                    "bbox": [
                        text_data['left'][i], text_data['top'][i],
                        text_data['left'][i] + text_data['width'][i],
                        text_data['top'][i] + text_data['height'][i]
                    ]
                }
                
                # Clasificar tipo de texto
                text_type = self._classify_text_type(text)
                element["type"] = text_type
                
                text_elements.append(element)
                
                if text_type == "electrical_label":
                    electrical_labels.append(element)
                elif text_type == "dimension":
                    dimensions.append(element)
                elif text_type == "note":
                    notes.append(element)
        
        return {
            "all_text": text_elements,
            "electrical_labels": electrical_labels,
            "dimensions": dimensions,
            "notes": notes,
            "summary": {
                "total_elements": len(text_elements),
                "electrical_count": len(electrical_labels),
                "dimension_count": len(dimensions),
                "note_count": len(notes)
            }
        }
    
    def _preprocess_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Preprocesa imagen para mejorar OCR"""
        # Redimensionar si es muy pequeña
        height, width = gray.shape
        if height < 300 or width < 300:
            scale_factor = max(300/height, 300/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Aplicar filtro de desenfoque para reducir ruido
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
    def _classify_text_type(self, text: str) -> str:
        """Clasifica el tipo de texto encontrado"""
        text_lower = text.lower()
        
        # Patrones para diferentes tipos
        dimension_pattern = r'^\d+[.,]?\d*\s*(mm|cm|m|in|ft|"|\')$'
        voltage_pattern = r'^\d+[.,]?\d*\s*(v|kv|mv|a|ma|ka|w|kw|mw|hz|khz)$'
        electrical_code_pattern = r'^[a-z]\d+$'
        
        if re.match(dimension_pattern, text_lower):
            return "dimension"
        elif re.match(voltage_pattern, text_lower):
            return "electrical_specification"
        elif re.match(electrical_code_pattern, text_lower):
            return "electrical_label"
        elif any(symbol in text_lower for symbol in ['panel', 'switch', 'outlet', 'light']):
            return "electrical_label"
        elif len(text) > 20:
            return "note"
        else:
            return "general_text"
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS AVANZADO DE COLORES Y FORMAS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def analyze_colors_and_materials(self, img_bgr: np.ndarray) -> Dict:
        """Análisis avanzado de colores con interpretación para planos eléctricos"""
        # Colores dominantes con K-means mejorado
        colors = self._extract_dominant_colors_advanced(img_bgr)
        
        # Análisis de distribución de colores
        color_distribution = self._analyze_color_distribution(img_bgr)
        
        # Interpretación para planos eléctricos
        electrical_interpretation = self._interpret_electrical_colors(colors)
        
        return {
            "dominant_colors": colors,
            "distribution": color_distribution,
            "electrical_interpretation": electrical_interpretation
        }
    
    def _extract_dominant_colors_advanced(self, img_bgr: np.ndarray, k: int = 8) -> List[Dict]:
        """Extrae colores dominantes con algoritmo K-means optimizado"""
        # Redimensionar para procesamiento rápido
        small = cv2.resize(img_bgr, (150, 150))
        data = small.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calcular porcentajes
        unique, counts = np.unique(labels, return_counts=True)
        centers = np.uint8(centers)
        
        colors = []
        for i, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            if percentage > 1:  # Filtrar colores minoritarios
                bgr = centers[i]
                colors.append({
                    "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],  # BGR a RGB
                    "percentage": round(percentage, 1),
                    "hex": f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                })
        
        return sorted(colors, key=lambda x: x["percentage"], reverse=True)
    
    def _analyze_color_distribution(self, img_bgr: np.ndarray) -> Dict:
        """Analiza la distribución espacial de colores"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Análisis por regiones de la imagen
        height, width = img_bgr.shape[:2]
        regions = {
            "top_left": img_bgr[0:height//2, 0:width//2],
            "top_right": img_bgr[0:height//2, width//2:width],
            "bottom_left": img_bgr[height//2:height, 0:width//2],
            "bottom_right": img_bgr[height//2:height, width//2:width]
        }
        
        region_analysis = {}
        for region_name, region_img in regions.items():
            mean_color = np.mean(region_img.reshape(-1, 3), axis=0)
            region_analysis[region_name] = {
                "mean_rgb": [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])],
                "brightness": int(np.mean(cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)))
            }
        
        return region_analysis
    
    def _interpret_electrical_colors(self, colors: List[Dict]) -> Dict:
        """Interpreta colores en el contexto de planos eléctricos"""
        interpretations = []
        
        for color in colors[:5]:  # Top 5 colores
            rgb = color["rgb"]
            r, g, b = rgb
            
            # Reglas de interpretación para planos eléctricos
            if r > 200 and g < 100 and b < 100:
                meaning = "Líneas de emergencia o alta tensión"
            elif r < 100 and g > 200 and b < 100:
                meaning = "Líneas de tierra o seguridad"
            elif r < 100 and g < 100 and b > 200:
                meaning = "Líneas de control o señalización"
            elif r > 200 and g > 200 and b < 100:
                meaning = "Advertencias o elementos especiales"
            elif r < 50 and g < 50 and b < 50:
                meaning = "Líneas principales o estructuras"
            elif r > 240 and g > 240 and b > 240:
                meaning = "Fondo del plano"
            else:
                meaning = "Elemento estándar del plano"
            
            interpretations.append({
                "color": rgb,
                "percentage": color["percentage"],
                "meaning": meaning
            })
        
        return interpretations
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # GENERACIÓN DE DESCRIPCIONES INTELIGENTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def generate_comprehensive_description(self, analysis_results: Dict) -> str:
        """Genera una descripción comprensiva y detallada del plano eléctrico"""
        sections = []
        
        # 1. Introducción
        sections.append("ANÁLISIS DETALLADO DEL PLANO ELÉCTRICO:")
        
        # 2. Símbolos detectados
        if analysis_results.get("symbols"):
            symbols = analysis_results["symbols"]
            symbol_types = {}
            for symbol in symbols:
                stype = symbol["type"]
                symbol_types[stype] = symbol_types.get(stype, 0) + 1
            
            if symbol_types:
                sections.append(f"\nCOMPONENTES IDENTIFICADOS ({len(symbols)} total):")
                for stype, count in sorted(symbol_types.items()):
                    sections.append(f"- {stype.replace('_', ' ').title()}: {count} unidad(es)")
        
        # 3. Análisis de rutas
        if analysis_results.get("routes") and isinstance(analysis_results["routes"], dict):
            routes = analysis_results["routes"]
            if routes.get("summary"):
                summary = routes["summary"]
                sections.append(f"\nSISTEMA DE CABLEADO:")
                sections.append(f"- Total de líneas detectadas: {summary['total_cables']}")
                sections.append(f"- Líneas horizontales: {summary['horizontal']}")
                sections.append(f"- Líneas verticales: {summary['vertical']}")
                sections.append(f"- Intersecciones: {summary['intersections']}")
        
        # 4. Información textual
        if analysis_results.get("text_analysis"):
            text_data = analysis_results["text_analysis"]
            if text_data.get("summary"):
                summary = text_data["summary"]
                sections.append(f"\nINFORMACIÓN TEXTUAL:")
                sections.append(f"- Etiquetas eléctricas: {summary['electrical_count']}")
                sections.append(f"- Dimensiones: {summary['dimension_count']}")
                sections.append(f"- Notas técnicas: {summary['note_count']}")
                
                # Incluir algunas etiquetas importantes
                if text_data.get("electrical_labels"):
                    important_labels = text_data["electrical_labels"][:5]
                    label_texts = [label["text"] for label in important_labels]
                    if label_texts:
                        sections.append(f"- Etiquetas principales: {', '.join(label_texts)}")
        
        # 5. Análisis de colores
        if analysis_results.get("color_analysis"):
            colors = analysis_results["color_analysis"]
            if colors.get("electrical_interpretation"):
                sections.append(f"\nANÁLISIS CROMÁTICO:")
                for interp in colors["electrical_interpretation"][:3]:
                    sections.append(f"- {interp['meaning']} ({interp['percentage']:.1f}%)")
        
        # 6. Evaluación general
        sections.append(f"\nEVALUACIÓN GENERAL:")
        complexity = self._assess_plan_complexity(analysis_results)
        sections.append(f"- Complejidad del plano: {complexity['level']}")
        sections.append(f"- Elementos totales detectados: {complexity['total_elements']}")
        sections.append(f"- Recomendación: {complexity['recommendation']}")
        
        return "\n".join(sections)
    
    def _assess_plan_complexity(self, analysis_results: Dict) -> Dict:
        """Evalúa la complejidad del plano eléctrico"""
        total_elements = 0
        
        # Contar elementos
        if analysis_results.get("symbols"):
            total_elements += len(analysis_results["symbols"])
        
        if analysis_results.get("routes") and isinstance(analysis_results["routes"], dict):
            if analysis_results["routes"].get("summary"):
                total_elements += analysis_results["routes"]["summary"]["total_cables"]
        
        if analysis_results.get("text_analysis") and analysis_results["text_analysis"].get("summary"):
            total_elements += analysis_results["text_analysis"]["summary"]["total_elements"]
        
        # Clasificar complejidad
        if total_elements < 20:
            level = "Básica"
            recommendation = "Plano simple, adecuado para instalaciones residenciales pequeñas"
        elif total_elements < 50:
            level = "Moderada"
            recommendation = "Plano de complejidad media, típico de instalaciones comerciales"
        elif total_elements < 100:
            level = "Alta"
            recommendation = "Plano complejo, requiere análisis detallado por profesionales"
        else:
            level = "Muy Alta"
            recommendation = "Plano muy complejo, instalación industrial o gran escala"
        
        return {
            "level": level,
            "total_elements": total_elements,
            "recommendation": recommendation
        }
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE UTILIDAD Y CLASIFICACIÓN
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def _map_yolo_to_electrical(self, yolo_class: str) -> str:
        """Mapea clases YOLO a componentes eléctricos"""
        mapping = {
            'person': 'interruptor',
            'chair': 'componente_electrico',
            'tvmonitor': 'panel_electrico',
            'laptop': 'unidad_control',
            'cell phone': 'dispositivo_sensor',
            'clock': 'temporizador',
            'stop sign': 'simbolo_seguridad',
            'car': 'motor_electrico',
            'truck': 'transformador',
            'bus': 'panel_distribucion'
        }
        return mapping.get(yolo_class, f'componente_{yolo_class}')
    
    def _classify_text_as_electrical(self, text: str) -> Optional[str]:
        """Clasifica texto como componente eléctrico"""
        text_lower = text.lower().strip()
        
        for electrical_type, keywords in self.electrical_symbols.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return electrical_type
        
        return None
    
    def _filter_and_classify_symbols(self, symbols: List[Dict]) -> List[Dict]:
        """Filtra duplicados y mejora clasificación de símbolos"""
        if not symbols:
            return []
        
        # Ordenar por confianza
        symbols.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Filtrar duplicados por proximidad
        filtered = []
        for symbol in symbols:
            is_duplicate = False
            for existing in filtered:
                if self._are_symbols_close(symbol, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(symbol)
        
        return filtered[:20]  # Limitar a 20 símbolos más relevantes
    
    def _are_symbols_close(self, symbol1: Dict, symbol2: Dict, threshold: int = 50) -> bool:
        """Determina si dos símbolos están muy cerca (posibles duplicados)"""
        bbox1 = symbol1["bbox"]
        bbox2 = symbol2["bbox"]
        
        center1 = [(bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2]
        center2 = [(bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2]
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance < threshold
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL DE ANÁLISIS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def analyze_electrical_plan(self, image_path: str) -> Dict:
        """Método principal para análisis completo de planos eléctricos"""
        try:
            # Cargar imagen
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": "No se pudo cargar la imagen"}
            
            print(f"🔍 Iniciando análisis avanzado de {image_path}")
            
            # Realizar todos los análisis
            results = {}
            
            # 1. Análisis de símbolos
            print("🔍 Detectando símbolos eléctricos...")
            results["symbols"] = self.detect_advanced_symbols(img_bgr)
            
            # 2. Análisis de rutas
            print("🔍 Analizando rutas y conexiones...")
            results["routes"] = self.detect_advanced_routes(img_bgr)
            
            # 3. Extracción de texto
            print("🔍 Extrayendo información textual...")
            results["text_analysis"] = self.extract_intelligent_text(img_bgr)
            
            # 4. Análisis de colores
            print("🔍 Analizando colores y materiales...")
            results["color_analysis"] = self.analyze_colors_and_materials(img_bgr)
            
            # 5. Generar descripción comprensiva
            print("📝 Generando descripción detallada...")
            results["comprehensive_description"] = self.generate_comprehensive_description(results)
            
            # 6. Resumen ejecutivo
            results["executive_summary"] = self._generate_executive_summary(results)
            
            print("✅ Análisis avanzado completado")
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis avanzado: {e}")
            return {"error": str(e)}
    
    def _generate_executive_summary(self, results: Dict) -> Dict:
        """Genera un resumen ejecutivo conciso"""
        summary = {
            "total_symbols": len(results.get("symbols", [])),
            "total_routes": 0,
            "total_text_elements": 0,
            "dominant_colors": [],
            "complexity_assessment": "No determinado",
            "key_findings": []
        }
        
        # Rutas
        if results.get("routes") and isinstance(results["routes"], dict):
            if results["routes"].get("summary"):
                summary["total_routes"] = results["routes"]["summary"]["total_cables"]
        
        # Texto
        if results.get("text_analysis") and results["text_analysis"].get("summary"):
            summary["total_text_elements"] = results["text_analysis"]["summary"]["total_elements"]
        
        # Colores dominantes
        if results.get("color_analysis") and results["color_analysis"].get("dominant_colors"):
            summary["dominant_colors"] = results["color_analysis"]["dominant_colors"][:3]
        
        # Hallazgos clave
        if summary["total_symbols"] > 10:
            summary["key_findings"].append("Plano con múltiples componentes eléctricos")
        if summary["total_routes"] > 20:
            summary["key_findings"].append("Sistema de cableado complejo")
        if summary["total_text_elements"] > 15:
            summary["key_findings"].append("Plano bien documentado con especificaciones")
        
        return summary

