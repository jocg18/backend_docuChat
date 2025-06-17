"""Analizador Universal de Imágenes Técnicas

Este servicio proporciona análisis avanzado para diversos tipos de documentos técnicos:
- Planos eléctricos
- Esquemas de postes y redes
- Diagramas de instalaciones
- Mapas técnicos y croquis
- Esquemas de procesos industriales
- Diagramas de flujo
- Planos arquitectónicos
- Diagramas de telecomunicaciones

Capacidades:
- Detección inteligente del tipo de documento
- Análisis específico por dominio
- Reconocimiento de símbolos gráficos universales
- Interpretación de rutas y conexiones
- Análisis de textos técnicos y especificaciones
- Generación de descripciones contextualizadas
"""

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image as PILImage
import re
from typing import Dict, List, Tuple, Optional, Any
import math
from enum import Enum
import uuid

class DocumentType(Enum):
    """Tipos de documentos técnicos detectables"""
    ELECTRICAL_PLAN = "electrical_plan"
    NETWORK_DIAGRAM = "network_diagram"
    INSTALLATION_SCHEME = "installation_scheme"
    TECHNICAL_MAP = "technical_map"
    PROCESS_DIAGRAM = "process_diagram"
    FLOW_CHART = "flow_chart"
    ARCHITECTURAL_PLAN = "architectural_plan"
    TELECOM_DIAGRAM = "telecom_diagram"
    MECHANICAL_DRAWING = "mechanical_drawing"
    INFRASTRUCTURE_PLAN = "infrastructure_plan"
    UNKNOWN = "unknown"

class UniversalTechnicalAnalyzer:
    def __init__(self):
        """Inicializa el analizador universal"""
        # Configuración de OCR optimizada
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=.,:;()[]{}"\'/\\°%'
        
        # Diccionarios de símbolos por dominio técnico
        self.technical_symbols = {
            # Símbolos eléctricos
            'electrical': {
                'switch': ['switch', 'SW', 'S', 'interruptor'],
                'outlet': ['outlet', 'receptacle', 'R', 'tomacorriente'],
                'light': ['light', 'lamp', 'L', 'luminaria'],
                'panel': ['panel', 'board', 'P', 'tablero'],
                'motor': ['motor', 'M'],
                'transformer': ['transformer', 'T', 'transformador'],
                'fuse': ['fuse', 'F', 'fusible'],
                'breaker': ['breaker', 'CB', 'disyuntor'],
                'ground': ['ground', 'GND', 'earth', 'tierra']
            },
            # Símbolos de red y telecomunicaciones
            'network': {
                'router': ['router', 'R', 'enrutador'],
                'switch_net': ['switch', 'SW', 'conmutador'],
                'server': ['server', 'SRV', 'servidor'],
                'firewall': ['firewall', 'FW', 'cortafuegos'],
                'antenna': ['antenna', 'ANT', 'antena'],
                'cable': ['cable', 'CAB', 'UTP', 'fiber'],
                'wifi': ['wifi', 'wireless', 'inalámbrico'],
                'modem': ['modem', 'MDM'],
                'repeater': ['repeater', 'REP', 'repetidor']
            },
            # Símbolos de instalaciones
            'installation': {
                'pipe': ['pipe', 'tubería', 'conducto'],
                'valve': ['valve', 'válvula', 'V'],
                'pump': ['pump', 'bomba', 'P'],
                'tank': ['tank', 'tanque', 'T'],
                'sensor': ['sensor', 'S', 'detector'],
                'meter': ['meter', 'medidor', 'M'],
                'filter': ['filter', 'filtro', 'F'],
                'heater': ['heater', 'calentador', 'H']
            },
            # Símbolos arquitectónicos
            'architectural': {
                'door': ['door', 'puerta', 'D'],
                'window': ['window', 'ventana', 'W'],
                'wall': ['wall', 'muro', 'pared'],
                'column': ['column', 'columna', 'C'],
                'beam': ['beam', 'viga', 'B'],
                'stair': ['stair', 'escalera', 'E'],
                'room': ['room', 'habitación', 'R']
            },
            # Símbolos de procesos
            'process': {
                'start': ['start', 'inicio', 'begin'],
                'end': ['end', 'fin', 'stop'],
                'decision': ['decision', 'decisión', 'if'],
                'process_step': ['process', 'proceso', 'step'],
                'data': ['data', 'datos', 'input'],
                'storage': ['storage', 'almacén', 'DB'],
                'connector': ['connector', 'conector']
            }
        }
        
        # Cargar modelo YOLOv5
        try:
            w_path = "./weights/yolov5s.pt"
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=w_path, source="github", trust_repo=True
            )
            self.model.conf = 0.20  # Umbral más bajo para mayor sensibilidad
            self.model.iou = 0.45
        except Exception as e:
            print(f"⚠️ Error cargando YOLOv5: {e}")
            self.model = None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DETECCIÓN INTELIGENTE DEL TIPO DE DOCUMENTO
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_document_type(self, img_bgr: np.ndarray, filename: str = "") -> DocumentType:
        """Detecta automáticamente el tipo de documento técnico"""
        print("🔍 Detectando tipo de documento técnico...")
        
        # Análisis por nombre de archivo
        filename_hints = self._analyze_filename(filename)
        
        # Análisis de contenido visual
        visual_hints = self._analyze_visual_content(img_bgr)
        
        # Análisis de texto
        text_hints = self._analyze_text_content(img_bgr)
        
        # Combinación de evidencias
        document_type = self._combine_detection_evidence(
            filename_hints, visual_hints, text_hints
        )
        
        print(f"📋 Tipo detectado: {document_type.value}")
        return document_type
    
    def _analyze_filename(self, filename: str) -> Dict[DocumentType, float]:
        """Analiza el nombre del archivo para detectar pistas del tipo"""
        hints = {doc_type: 0.0 for doc_type in DocumentType}
        
        filename_lower = filename.lower()
        
        # Patrones por tipo de documento
        patterns = {
            DocumentType.ELECTRICAL_PLAN: ['electr', 'electric', 'power', 'energia', 'instalacion'],
            DocumentType.NETWORK_DIAGRAM: ['network', 'red', 'topology', 'lan', 'wan', 'wifi'],
            DocumentType.INSTALLATION_SCHEME: ['instalacion', 'installation', 'esquema', 'scheme'],
            DocumentType.TECHNICAL_MAP: ['map', 'mapa', 'layout', 'planta', 'ubicacion'],
            DocumentType.PROCESS_DIAGRAM: ['process', 'proceso', 'flow', 'flujo', 'workflow'],
            DocumentType.ARCHITECTURAL_PLAN: ['arquitectura', 'architectural', 'building', 'edificio'],
            DocumentType.TELECOM_DIAGRAM: ['telecom', 'comunicacion', 'antenna', 'signal'],
            DocumentType.MECHANICAL_DRAWING: ['mechanical', 'mecanico', 'machine', 'maquina'],
            DocumentType.INFRASTRUCTURE_PLAN: ['infrastructure', 'infraestructura', 'civil', 'urbano']
        }
        
        for doc_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    hints[doc_type] += 0.3
        
        return hints
    
    def _analyze_visual_content(self, img_bgr: np.ndarray) -> Dict[DocumentType, float]:
        """Analiza el contenido visual para detectar patrones específicos"""
        hints = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Análisis de características visuales
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de líneas (importante para todos los tipos técnicos)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            line_count = len(lines)
            
            # Análisis de orientación de líneas
            horizontal_lines = 0
            vertical_lines = 0
            diagonal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                
                if -15 <= angle <= 15 or 165 <= abs(angle) <= 180:
                    horizontal_lines += 1
                elif 75 <= abs(angle) <= 105:
                    vertical_lines += 1
                else:
                    diagonal_lines += 1
            
            # Patrones por tipo de documento
            if horizontal_lines > vertical_lines * 2:  # Muchas líneas horizontales
                hints[DocumentType.PROCESS_DIAGRAM] += 0.2
                hints[DocumentType.FLOW_CHART] += 0.2
            
            if vertical_lines > horizontal_lines * 1.5:  # Muchas líneas verticales
                hints[DocumentType.ELECTRICAL_PLAN] += 0.2
                hints[DocumentType.NETWORK_DIAGRAM] += 0.1
            
            if line_count > 50:  # Muchas líneas = plano complejo
                hints[DocumentType.ELECTRICAL_PLAN] += 0.3
                hints[DocumentType.ARCHITECTURAL_PLAN] += 0.2
                hints[DocumentType.MECHANICAL_DRAWING] += 0.2
        
        # Detección de círculos (símbolos técnicos)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=100)
        
        if circles is not None:
            circle_count = len(circles[0])
            if circle_count > 5:
                hints[DocumentType.ELECTRICAL_PLAN] += 0.3
                hints[DocumentType.PROCESS_DIAGRAM] += 0.2
                hints[DocumentType.NETWORK_DIAGRAM] += 0.2
        
        # Análisis de color para detectar códigos técnicos
        color_variance = np.var(img_bgr)
        if color_variance > 1000:  # Imagen con muchos colores
            hints[DocumentType.NETWORK_DIAGRAM] += 0.2
            hints[DocumentType.PROCESS_DIAGRAM] += 0.1
        
        return hints
    
    def _analyze_text_content(self, img_bgr: np.ndarray) -> Dict[DocumentType, float]:
        """Analiza el contenido textual para identificar el dominio"""
        hints = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Extraer texto
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=self.tesseract_config).lower()
        
        # Palabras clave por dominio
        domain_keywords = {
            DocumentType.ELECTRICAL_PLAN: [
                'volt', 'amp', 'watt', 'electrical', 'electrico', 'power', 'energia',
                'switch', 'outlet', 'panel', 'circuit', 'wire', 'cable', 'phase'
            ],
            DocumentType.NETWORK_DIAGRAM: [
                'network', 'ethernet', 'wifi', 'router', 'switch', 'server', 'ip',
                'lan', 'wan', 'tcp', 'udp', 'firewall', 'gateway', 'subnet'
            ],
            DocumentType.INSTALLATION_SCHEME: [
                'installation', 'instalacion', 'pipe', 'valve', 'pump', 'tank',
                'pressure', 'flow', 'temperature', 'sensor', 'control'
            ],
            DocumentType.PROCESS_DIAGRAM: [
                'process', 'proceso', 'start', 'end', 'decision', 'workflow',
                'step', 'stage', 'input', 'output', 'data', 'control'
            ],
            DocumentType.ARCHITECTURAL_PLAN: [
                'room', 'door', 'window', 'wall', 'floor', 'ceiling', 'stairs',
                'kitchen', 'bathroom', 'bedroom', 'scale', 'dimension'
            ],
            DocumentType.MECHANICAL_DRAWING: [
                'bearing', 'shaft', 'gear', 'bolt', 'tolerance', 'dimension',
                'assembly', 'part', 'material', 'surface', 'finish'
            ]
        }
        
        for doc_type, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    hints[doc_type] += 0.2
        
        return hints
    
    def _combine_detection_evidence(self, filename_hints: Dict, visual_hints: Dict, text_hints: Dict) -> DocumentType:
        """Combina todas las evidencias para determinar el tipo de documento"""
        combined_scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Pesos para cada tipo de evidencia
        weights = {
            'filename': 0.3,
            'visual': 0.4,
            'text': 0.3
        }
        
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
                
            combined_scores[doc_type] = (
                filename_hints.get(doc_type, 0) * weights['filename'] +
                visual_hints.get(doc_type, 0) * weights['visual'] +
                text_hints.get(doc_type, 0) * weights['text']
            )
        
        # Encontrar el tipo con mayor puntuación
        best_type = max(combined_scores, key=combined_scores.get)
        best_score = combined_scores[best_type]
        
        # Si la puntuación es muy baja, marcar como desconocido
        if best_score < 0.3:
            return DocumentType.UNKNOWN
        
        return best_type
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS UNIVERSAL DE SÍMBOLOS TÉCNICOS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_universal_symbols(self, img_bgr: np.ndarray, document_type: DocumentType) -> List[Dict]:
        """Detecta símbolos técnicos específicos del tipo de documento"""
        print(f"🔍 Detectando símbolos para {document_type.value}...")
        
        symbols = []
        
        # Detección con YOLOv5 (general)
        if self.model is not None:
            yolo_symbols = self._detect_yolo_symbols(img_bgr, document_type)
            symbols.extend(yolo_symbols)
        
        # Detección específica por tipo de documento
        if document_type == DocumentType.ELECTRICAL_PLAN:
            electrical_symbols = self._detect_electrical_symbols(img_bgr)
            symbols.extend(electrical_symbols)
        elif document_type == DocumentType.NETWORK_DIAGRAM:
            network_symbols = self._detect_network_symbols(img_bgr)
            symbols.extend(network_symbols)
        elif document_type == DocumentType.PROCESS_DIAGRAM:
            process_symbols = self._detect_process_symbols(img_bgr)
            symbols.extend(process_symbols)
        elif document_type == DocumentType.ARCHITECTURAL_PLAN:
            arch_symbols = self._detect_architectural_symbols(img_bgr)
            symbols.extend(arch_symbols)
        else:
            # Detección genérica para tipos no específicos
            generic_symbols = self._detect_generic_technical_symbols(img_bgr)
            symbols.extend(generic_symbols)
        
        # Detección por texto
        text_symbols = self._detect_text_based_symbols(img_bgr, document_type)
        symbols.extend(text_symbols)
        
        # Filtrar duplicados
        symbols = self._filter_duplicate_symbols(symbols)
        
        print(f"✅ Detectados {len(symbols)} símbolos técnicos")
        return symbols
    
    def _detect_yolo_symbols(self, img_bgr: np.ndarray, document_type: DocumentType) -> List[Dict]:
        """Detección con YOLOv5 adaptada al tipo de documento"""
        symbols = []
        try:
            results = self.model(img_bgr)
            
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > 0.20:
                    yolo_class = self.model.names[int(cls)]
                    technical_type = self._map_yolo_to_technical(yolo_class, document_type)
                    
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    
                    symbols.append({
                        "type": technical_type,
                        "subtype": yolo_class,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "area": area,
                        "detection_method": "yolo",
                        "domain": document_type.value
                    })
        except Exception as e:
            print(f"⚠️ Error en detección YOLO: {e}")
        
        return symbols
    
    def _map_yolo_to_technical(self, yolo_class: str, document_type: DocumentType) -> str:
        """Mapea clases YOLO a símbolos técnicos según el tipo de documento"""
        # Mapeos específicos por tipo de documento
        mappings = {
            DocumentType.ELECTRICAL_PLAN: {
                'person': 'interruptor',
                'chair': 'componente_electrico',
                'tvmonitor': 'panel_electrico',
                'laptop': 'unidad_control',
                'cell phone': 'dispositivo_sensor',
                'clock': 'temporizador',
                'car': 'motor_electrico',
                'truck': 'transformador'
            },
            DocumentType.NETWORK_DIAGRAM: {
                'laptop': 'computadora',
                'tvmonitor': 'servidor',
                'cell phone': 'dispositivo_movil',
                'keyboard': 'terminal',
                'mouse': 'dispositivo_entrada',
                'book': 'documentacion',
                'chair': 'estacion_trabajo'
            },
            DocumentType.PROCESS_DIAGRAM: {
                'stop sign': 'decision_point',
                'traffic light': 'control_process',
                'car': 'transport_step',
                'truck': 'heavy_process',
                'person': 'human_intervention',
                'clock': 'time_control'
            },
            DocumentType.ARCHITECTURAL_PLAN: {
                'door': 'puerta',
                'chair': 'mobiliario',
                'bed': 'cama',
                'dining table': 'mesa',
                'toilet': 'sanitario',
                'tv': 'electrodomestico'
            }
        }
        
        domain_mapping = mappings.get(document_type, {})
        return domain_mapping.get(yolo_class, f'elemento_{yolo_class}')
    
    def _detect_electrical_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección específica de símbolos eléctricos"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de círculos (conexiones, luminarias)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=3, maxRadius=30)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                symbols.append({
                    "type": "conexion_electrica",
                    "subtype": "junction_point",
                    "confidence": 0.8,
                    "bbox": [x-r, y-r, x+r, y+r],
                    "area": int(math.pi * r * r),
                    "detection_method": "electrical_geometric",
                    "domain": "electrical"
                })
        
        return symbols
    
    def _detect_network_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección específica de símbolos de red"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de rectángulos (dispositivos de red)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                
                if len(approx) == 4 and 0.5 < w/h < 2.0:
                    symbols.append({
                        "type": "dispositivo_red",
                        "subtype": "network_device",
                        "confidence": 0.7,
                        "bbox": [x, y, x+w, y+h],
                        "area": int(area),
                        "detection_method": "network_geometric",
                        "domain": "network"
                    })
        
        return symbols
    
    def _detect_process_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección específica de símbolos de proceso"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de formas de diagrama de flujo
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 150 < area < 10000:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                
                # Clasificar por número de vértices
                if len(approx) == 4:
                    ar = w / float(h)
                    if 0.8 < ar < 1.2:
                        symbol_type = "decision_diamond" if len(approx) == 4 else "process_rectangle"
                    else:
                        symbol_type = "process_rectangle"
                elif len(approx) > 8:
                    symbol_type = "terminal_oval"
                else:
                    symbol_type = "connector_polygon"
                
                symbols.append({
                    "type": symbol_type,
                    "subtype": "flow_element",
                    "confidence": 0.75,
                    "bbox": [x, y, x+w, y+h],
                    "area": int(area),
                    "detection_method": "process_geometric",
                    "domain": "process"
                })
        
        return symbols
    
    def _detect_architectural_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección específica de símbolos arquitectónicos"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de líneas para muros
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Agrupar líneas paralelas (muros)
            parallel_groups = self._group_parallel_lines(lines)
            
            for group in parallel_groups:
                if len(group) >= 2:  # Dos líneas paralelas = posible muro
                    symbols.append({
                        "type": "muro_arquitectonico",
                        "subtype": "wall_structure",
                        "confidence": 0.8,
                        "bbox": self._get_group_bbox(group),
                        "area": self._calculate_group_area(group),
                        "detection_method": "architectural_geometric",
                        "domain": "architectural"
                    })
        
        return symbols
    
    def _detect_generic_technical_symbols(self, img_bgr: np.ndarray) -> List[Dict]:
        """Detección genérica para documentos técnicos no específicos"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de formas geométricas básicas
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 8000:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                
                # Clasificación básica
                if len(approx) == 3:
                    symbol_type = "triangular_element"
                elif len(approx) == 4:
                    symbol_type = "rectangular_element"
                elif len(approx) > 8:
                    symbol_type = "circular_element"
                else:
                    symbol_type = "polygonal_element"
                
                symbols.append({
                    "type": symbol_type,
                    "subtype": "generic_technical",
                    "confidence": 0.6,
                    "bbox": [x, y, x+w, y+h],
                    "area": int(area),
                    "detection_method": "generic_geometric",
                    "domain": "generic"
                })
        
        return symbols
    
    def _detect_text_based_symbols(self, img_bgr: np.ndarray, document_type: DocumentType) -> List[Dict]:
        """Detección de símbolos basada en texto según el tipo de documento"""
        symbols = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Extraer texto con información de posición
        text_data = pytesseract.image_to_data(
            gray, config=self.tesseract_config, output_type=pytesseract.Output.DICT
        )
        
        # Símbolos específicos del dominio
        domain_symbols = self.technical_symbols.get(
            document_type.value.split('_')[0], self.technical_symbols.get('electrical', {})
        )
        
        for i, text in enumerate(text_data['text']):
            if text.strip() and int(text_data['conf'][i]) > 30:
                # Buscar coincidencias con símbolos del dominio
                technical_type = self._classify_text_as_technical(text, domain_symbols)
                
                if technical_type:
                    x, y, w, h = (
                        text_data['left'][i], text_data['top'][i],
                        text_data['width'][i], text_data['height'][i]
                    )
                    
                    symbols.append({
                        "type": technical_type,
                        "subtype": "text_identified",
                        "confidence": int(text_data['conf'][i]) / 100.0,
                        "bbox": [x, y, x+w, y+h],
                        "area": w * h,
                        "text": text,
                        "detection_method": "text_recognition",
                        "domain": document_type.value
                    })
        
        return symbols
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS UNIVERSAL DE RUTAS Y CONEXIONES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_universal_routes(self, img_bgr: np.ndarray, document_type: DocumentType) -> Dict:
        """Detecta rutas y conexiones específicas del tipo de documento"""
        print(f"🔍 Analizando rutas para {document_type.value}...")
        
        if document_type == DocumentType.ELECTRICAL_PLAN:
            return self._detect_electrical_routes(img_bgr)
        elif document_type == DocumentType.NETWORK_DIAGRAM:
            return self._detect_network_routes(img_bgr)
        elif document_type == DocumentType.PROCESS_DIAGRAM:
            return self._detect_process_flow(img_bgr)
        elif document_type == DocumentType.INSTALLATION_SCHEME:
            return self._detect_installation_pipes(img_bgr)
        else:
            return self._detect_generic_connections(img_bgr)
    
    def _detect_electrical_routes(self, img_bgr: np.ndarray) -> Dict:
        """Detecta rutas eléctricas específicas"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=15, maxLineGap=8)
        
        routes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                
                # Clasificar tipo de cable
                if length > 50:  # Solo líneas significativas
                    if -15 <= angle <= 15 or 165 <= abs(angle) <= 180:
                        route_type = "cable_horizontal"
                    elif 75 <= abs(angle) <= 105:
                        route_type = "cable_vertical"
                    else:
                        route_type = "cable_diagonal"
                    
                    routes.append({
                        "type": route_type,
                        "start": [x1, y1],
                        "end": [x2, y2],
                        "length": length,
                        "angle": angle,
                        "domain": "electrical"
                    })
        
        # Detectar intersecciones
        intersections = self._detect_intersections(routes)
        
        return {
            "routes": routes[:15],  # Limitar para evitar ruido
            "intersections": intersections,
            "summary": {
                "total_routes": len(routes),
                "horizontal": len([r for r in routes if r["type"] == "cable_horizontal"]),
                "vertical": len([r for r in routes if r["type"] == "cable_vertical"]),
                "intersections": len(intersections),
                "route_type": "electrical_wiring"
            }
        }
    
    def _detect_network_routes(self, img_bgr: np.ndarray) -> Dict:
        """Detecta conexiones de red"""
        # Similar a electrical pero con diferentes interpretaciones
        base_routes = self._detect_electrical_routes(img_bgr)
        
        # Reinterpretar para contexto de red
        for route in base_routes["routes"]:
            route["type"] = route["type"].replace("cable", "conexion_red")
            route["domain"] = "network"
        
        base_routes["summary"]["route_type"] = "network_connections"
        return base_routes
    
    def _detect_process_flow(self, img_bgr: np.ndarray) -> Dict:
        """Detecta flujos de proceso"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detectar flechas y direcciones
        # Implementación simplificada
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        
        flows = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 30:
                    flows.append({
                        "type": "process_flow",
                        "start": [x1, y1],
                        "end": [x2, y2],
                        "length": length,
                        "direction": "forward",  # Simplificado
                        "domain": "process"
                    })
        
        return {
            "routes": flows[:10],
            "intersections": [],
            "summary": {
                "total_routes": len(flows),
                "route_type": "process_flows",
                "flow_direction": "mixed"
            }
        }
    
    def _detect_installation_pipes(self, img_bgr: np.ndarray) -> Dict:
        """Detecta tuberías y conductos"""
        # Similar a electrical pero interpretado como tuberías
        base_routes = self._detect_electrical_routes(img_bgr)
        
        for route in base_routes["routes"]:
            route["type"] = route["type"].replace("cable", "tuberia")
            route["domain"] = "installation"
        
        base_routes["summary"]["route_type"] = "pipe_system"
        return base_routes
    
    def _detect_generic_connections(self, img_bgr: np.ndarray) -> Dict:
        """Detecta conexiones genéricas"""
        base_routes = self._detect_electrical_routes(img_bgr)
        
        for route in base_routes["routes"]:
            route["type"] = "conexion_generica"
            route["domain"] = "generic"
        
        base_routes["summary"]["route_type"] = "generic_connections"
        return base_routes
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE UTILIDAD
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def _filter_duplicate_symbols(self, symbols: List[Dict]) -> List[Dict]:
        """Filtra símbolos duplicados por proximidad"""
        if not symbols:
            return []
        
        symbols.sort(key=lambda x: x["confidence"], reverse=True)
        
        filtered = []
        for symbol in symbols:
            is_duplicate = False
            for existing in filtered:
                if self._symbols_overlap(symbol, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(symbol)
        
        return filtered[:25]  # Limitar a 25 símbolos más relevantes
    
    def _symbols_overlap(self, symbol1: Dict, symbol2: Dict, threshold: int = 40) -> bool:
        """Determina si dos símbolos se superponen"""
        bbox1 = symbol1["bbox"]
        bbox2 = symbol2["bbox"]
        
        center1 = [(bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2]
        center2 = [(bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2]
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < threshold
    
    def _classify_text_as_technical(self, text: str, domain_symbols: Dict) -> Optional[str]:
        """Clasifica texto como símbolo técnico del dominio"""
        text_lower = text.lower().strip()
        
        for technical_type, keywords in domain_symbols.items():
            for keyword in keywords:
                if keyword.lower() in text_lower or text_lower in keyword.lower():
                    return technical_type
        
        return None
    
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
                        "type": "intersection"
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
    
    def _group_parallel_lines(self, lines) -> List[List]:
        """Agrupa líneas paralelas"""
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            x1, y1, x2, y2 = line1[0]
            angle1 = math.atan2(y2-y1, x2-x1)
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                angle2 = math.atan2(y4-y3, x4-x3)
                
                # Si las líneas son aproximadamente paralelas
                if abs(angle1 - angle2) < 0.2 or abs(abs(angle1 - angle2) - math.pi) < 0.2:
                    group.append(line2)
                    used.add(j)
            
            if len(group) >= 1:
                groups.append(group)
        
        return groups
    
    def _get_group_bbox(self, group) -> List[int]:
        """Obtiene bounding box de un grupo de líneas"""
        all_points = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            all_points.extend([(x1, y1), (x2, y2)])
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def _calculate_group_area(self, group) -> int:
        """Calcula área aproximada de un grupo"""
        bbox = self._get_group_bbox(group)
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS COMPLETO Y GENERACIÓN DE DESCRIPCIONES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def analyze_technical_document(self, image_path: str) -> Dict:
        """Método principal para análisis completo de documentos técnicos universales"""
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": "No se pudo cargar la imagen"}
            
            filename = image_path.split('/')[-1] if '/' in image_path else image_path.split('\\')[-1]
            print(f"🔍 Analizando documento técnico: {filename}")
            
            # 1. Detectar tipo de documento
            document_type = self.detect_document_type(img_bgr, filename)
            
            # 2. Análisis específico por tipo
            results = {
                "document_type": document_type.value,
                "filename": filename,
                "analysis_timestamp": str(uuid.uuid4())
            }
            
            # 3. Detectar símbolos técnicos
            print("🔍 Detectando símbolos técnicos...")
            results["symbols"] = self.detect_universal_symbols(img_bgr, document_type)
            
            # 4. Analizar rutas y conexiones
            print("🔍 Analizando rutas y conexiones...")
            results["routes"] = self.detect_universal_routes(img_bgr, document_type)
            
            # 5. Extraer y clasificar texto
            print("🔍 Extrayendo información textual...")
            results["text_analysis"] = self._extract_technical_text(img_bgr, document_type)
            
            # 6. Analizar colores y materiales
            print("🔍 Analizando colores y elementos visuales...")
            results["visual_analysis"] = self._analyze_technical_visuals(img_bgr, document_type)
            
            # 7. Generar descripción comprensiva
            print("📝 Generando descripción técnica...")
            results["technical_description"] = self._generate_technical_description(results)
            
            # 8. Evaluación y clasificación
            results["technical_assessment"] = self._assess_technical_complexity(results)
            
            print("✅ Análisis técnico universal completado")
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis técnico: {e}")
            return {"error": str(e)}
    
    def _extract_technical_text(self, img_bgr: np.ndarray, document_type: DocumentType) -> Dict:
        """Extrae y clasifica texto técnico específico"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Mejorar imagen para OCR
        processed = self._preprocess_for_technical_ocr(gray)
        
        # Extraer texto
        text_data = pytesseract.image_to_data(
            processed, config=self.tesseract_config, output_type=pytesseract.Output.DICT
        )
        
        # Clasificar por tipo de documento
        text_elements = []
        technical_specs = []
        labels = []
        dimensions = []
        notes = []
        
        for i, text in enumerate(text_data['text']):
            if text.strip() and int(text_data['conf'][i]) > 25:
                element = {
                    "text": text.strip(),
                    "confidence": int(text_data['conf'][i]),
                    "bbox": [
                        text_data['left'][i], text_data['top'][i],
                        text_data['left'][i] + text_data['width'][i],
                        text_data['top'][i] + text_data['height'][i]
                    ]
                }
                
                # Clasificar según el tipo de documento
                text_type = self._classify_technical_text(text, document_type)
                element["type"] = text_type
                
                text_elements.append(element)
                
                if text_type == "technical_specification":
                    technical_specs.append(element)
                elif text_type == "label":
                    labels.append(element)
                elif text_type == "dimension":
                    dimensions.append(element)
                elif text_type == "note":
                    notes.append(element)
        
        return {
            "all_text": text_elements,
            "technical_specifications": technical_specs,
            "labels": labels,
            "dimensions": dimensions,
            "notes": notes,
            "summary": {
                "total_elements": len(text_elements),
                "specs_count": len(technical_specs),
                "labels_count": len(labels),
                "dimensions_count": len(dimensions),
                "notes_count": len(notes)
            }
        }
    
    def _preprocess_for_technical_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Preprocesa imagen para OCR técnico mejorado"""
        # Redimensionar si es necesario
        height, width = gray.shape
        if height < 400 or width < 400:
            scale = max(400/height, 400/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Mejorar contraste para texto técnico
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reducir ruido preservando texto
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def _classify_technical_text(self, text: str, document_type: DocumentType) -> str:
        """Clasifica texto según el contexto técnico"""
        text_lower = text.lower().strip()
        
        # Patrones específicos por tipo de documento
        if document_type == DocumentType.ELECTRICAL_PLAN:
            if re.match(r'^\d+[.,]?\d*\s*(v|kv|mv|a|ma|ka|w|kw|mw|hz|khz)$', text_lower):
                return "technical_specification"
        elif document_type == DocumentType.NETWORK_DIAGRAM:
            if re.match(r'^\d+[.,]?\d*\s*(mbps|gbps|ghz|mhz)$', text_lower):
                return "technical_specification"
            elif 'ip' in text_lower or re.match(r'^\d+\.\d+\.\d+\.\d+$', text):
                return "technical_specification"
        elif document_type == DocumentType.PROCESS_DIAGRAM:
            if any(word in text_lower for word in ['°c', 'bar', 'psi', 'l/min', 'm³/h']):
                return "technical_specification"
        
        # Patrones generales
        if re.match(r'^\d+[.,]?\d*\s*(mm|cm|m|in|ft|"|\')$', text_lower):
            return "dimension"
        elif re.match(r'^[a-z]\d+$', text_lower) or len(text) <= 5:
            return "label"
        elif len(text) > 20:
            return "note"
        else:
            return "general_text"
    
    def _analyze_technical_visuals(self, img_bgr: np.ndarray, document_type: DocumentType) -> Dict:
        """Analiza elementos visuales específicos del dominio técnico"""
        # Análisis de colores con interpretación técnica
        colors = self._extract_technical_colors(img_bgr, document_type)
        
        # Análisis de patrones visuales
        patterns = self._detect_visual_patterns(img_bgr, document_type)
        
        # Análisis de escalas y proporciones
        scale_analysis = self._analyze_scale_indicators(img_bgr)
        
        return {
            "dominant_colors": colors,
            "visual_patterns": patterns,
            "scale_analysis": scale_analysis,
            "technical_interpretation": self._interpret_technical_visuals(colors, document_type)
        }
    
    def _extract_technical_colors(self, img_bgr: np.ndarray, document_type: DocumentType) -> List[Dict]:
        """Extrae colores con interpretación técnica"""
        # Análisis K-means de colores
        small = cv2.resize(img_bgr, (150, 150))
        data = small.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        centers = np.uint8(centers)
        
        colors = []
        for i, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            if percentage > 2:  # Solo colores significativos
                bgr = centers[i]
                colors.append({
                    "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
                    "percentage": round(percentage, 1),
                    "hex": f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}",
                    "technical_meaning": self._interpret_color_technically(bgr, document_type)
                })
        
        return sorted(colors, key=lambda x: x["percentage"], reverse=True)
    
    def _interpret_color_technically(self, bgr: np.ndarray, document_type: DocumentType) -> str:
        """Interpreta colores en contexto técnico específico"""
        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
        
        # Interpretaciones por tipo de documento
        if document_type == DocumentType.ELECTRICAL_PLAN:
            if r > 200 and g < 100 and b < 100:
                return "Líneas de alta tensión o emergencia"
            elif r < 100 and g > 200 and b < 100:
                return "Líneas de tierra o seguridad"
            elif r < 100 and g < 100 and b > 200:
                return "Líneas de control o señalización"
        elif document_type == DocumentType.NETWORK_DIAGRAM:
            if r < 100 and g < 100 and b > 200:
                return "Conexiones de datos o fibra óptica"
            elif r > 200 and g > 100 and b < 100:
                return "Conexiones de energía o críticas"
        elif document_type == DocumentType.PROCESS_DIAGRAM:
            if r > 200 and g < 100 and b < 100:
                return "Flujo de alta temperatura o peligro"
            elif r < 100 and g < 100 and b > 200:
                return "Flujo de baja temperatura o agua"
        
        # Interpretación genérica
        if r < 50 and g < 50 and b < 50:
            return "Elementos estructurales principales"
        elif r > 240 and g > 240 and b > 240:
            return "Fondo o espacio libre"
        else:
            return "Elemento técnico estándar"
    
    def _detect_visual_patterns(self, img_bgr: np.ndarray, document_type: DocumentType) -> Dict:
        """Detecta patrones visuales específicos"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detección de patrones de líneas
        edges = cv2.Canny(gray, 50, 150)
        line_density = np.sum(edges > 0) / edges.size
        
        # Detección de patrones de formas
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_complexity = len(contours)
        
        # Análisis de simetría
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        if right_half.shape[1] == left_half.shape[1]:
            symmetry_score = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
        else:
            symmetry_score = 0.0
        
        return {
            "line_density": float(line_density),
            "shape_complexity": shape_complexity,
            "symmetry_score": float(symmetry_score),
            "pattern_type": self._classify_visual_pattern(line_density, shape_complexity, symmetry_score)
        }
    
    def _classify_visual_pattern(self, line_density: float, shape_complexity: int, symmetry: float) -> str:
        """Clasifica el patrón visual del documento"""
        if line_density > 0.3 and shape_complexity > 50:
            return "highly_detailed_technical"
        elif symmetry > 0.7:
            return "symmetric_design"
        elif line_density > 0.2:
            return "line_based_diagram"
        elif shape_complexity > 20:
            return "shape_based_diagram"
        else:
            return "simple_technical"
    
    def _analyze_scale_indicators(self, img_bgr: np.ndarray) -> Dict:
        """Analiza indicadores de escala en el documento"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Buscar líneas de escala o reglas
        text = pytesseract.image_to_string(gray, config=self.tesseract_config).lower()
        
        scale_indicators = []
        scale_patterns = [
            r'1:(\d+)',
            r'scale\s+1:(\d+)',
            r'escala\s+1:(\d+)',
            r'(\d+)\s*mm',
            r'(\d+)\s*cm',
            r'(\d+)\s*m'
        ]
        
        for pattern in scale_patterns:
            matches = re.findall(pattern, text)
            if matches:
                scale_indicators.extend(matches)
        
        return {
            "scale_found": len(scale_indicators) > 0,
            "scale_indicators": scale_indicators,
            "estimated_scale": scale_indicators[0] if scale_indicators else None
        }
    
    def _interpret_technical_visuals(self, colors: List[Dict], document_type: DocumentType) -> List[str]:
        """Genera interpretaciones técnicas de los elementos visuales"""
        interpretations = []
        
        # Interpretaciones basadas en colores dominantes
        for color in colors[:3]:
            meaning = color.get("technical_meaning", "")
            if meaning and meaning != "Elemento técnico estándar":
                interpretations.append(
                    f"{meaning} ({color['percentage']:.1f}% del documento)"
                )
        
        return interpretations
    
    def _generate_technical_description(self, analysis_results: Dict) -> str:
        """Genera descripción técnica comprensiva"""
        doc_type = analysis_results.get("document_type", "unknown")
        symbols = analysis_results.get("symbols", [])
        routes = analysis_results.get("routes", {})
        text_analysis = analysis_results.get("text_analysis", {})
        visual_analysis = analysis_results.get("visual_analysis", {})
        
        # Determinar dominio técnico
        domain_names = {
            "electrical_plan": "plano eléctrico",
            "network_diagram": "diagrama de red",
            "installation_scheme": "esquema de instalación",
            "process_diagram": "diagrama de proceso",
            "architectural_plan": "plano arquitectónico",
            "mechanical_drawing": "dibujo mecánico",
            "technical_map": "mapa técnico",
            "unknown": "documento técnico"
        }
        
        domain_name = domain_names.get(doc_type, "documento técnico")
        
        sections = []
        sections.append(f"ANÁLISIS TÉCNICO DE {domain_name.upper()}")
        
        # Resumen de componentes
        if symbols:
            symbol_types = {}
            for symbol in symbols:
                stype = symbol["type"]
                symbol_types[stype] = symbol_types.get(stype, 0) + 1
            
            sections.append(f"\nCOMPONENTES TÉCNICOS IDENTIFICADOS ({len(symbols)} total):")
            for stype, count in sorted(symbol_types.items()):
                sections.append(f"- {stype.replace('_', ' ').title()}: {count} elemento(s)")
        
        # Análisis de conexiones
        if routes and isinstance(routes, dict) and routes.get("summary"):
            summary = routes["summary"]
            sections.append(f"\nSISTEMA DE CONEXIONES:")
            sections.append(f"- Tipo: {summary.get('route_type', 'genérico')}")
            sections.append(f"- Total de conexiones: {summary.get('total_routes', 0)}")
            if summary.get('intersections', 0) > 0:
                sections.append(f"- Puntos de intersección: {summary['intersections']}")
        
        # Información textual
        if text_analysis and text_analysis.get("summary"):
            summary = text_analysis["summary"]
            sections.append(f"\nINFORMACIÓN TÉCNICA TEXTUAL:")
            sections.append(f"- Especificaciones técnicas: {summary.get('specs_count', 0)}")
            sections.append(f"- Etiquetas identificadas: {summary.get('labels_count', 0)}")
            sections.append(f"- Dimensiones encontradas: {summary.get('dimensions_count', 0)}")
        
        # Análisis visual
        if visual_analysis:
            patterns = visual_analysis.get("visual_patterns", {})
            pattern_type = patterns.get("pattern_type", "unknown")
            
            pattern_descriptions = {
                "highly_detailed_technical": "Documento técnico altamente detallado",
                "symmetric_design": "Diseño simétrico y estructurado",
                "line_based_diagram": "Diagrama basado en líneas y conexiones",
                "shape_based_diagram": "Diagrama basado en formas y símbolos",
                "simple_technical": "Documento técnico simple"
            }
            
            sections.append(f"\nCARACTERÍSTICAS VISUALES:")
            sections.append(f"- Patrón visual: {pattern_descriptions.get(pattern_type, 'Patrón técnico estándar')}")
            
            # Interpretaciones técnicas de colores
            tech_interpretations = visual_analysis.get("technical_interpretation", [])
            if tech_interpretations:
                sections.append("- Interpretación cromática:")
                for interpretation in tech_interpretations[:3]:
                    sections.append(f"  • {interpretation}")
        
        return "\n".join(sections)
    
    def _assess_technical_complexity(self, analysis_results: Dict) -> Dict:
        """Evalúa la complejidad técnica del documento"""
        symbols_count = len(analysis_results.get("symbols", []))
        routes_count = analysis_results.get("routes", {}).get("summary", {}).get("total_routes", 0)
        text_elements = analysis_results.get("text_analysis", {}).get("summary", {}).get("total_elements", 0)
        
        total_elements = symbols_count + routes_count + text_elements
        
        # Clasificación de complejidad específica por dominio
        doc_type = analysis_results.get("document_type", "unknown")
        
        if doc_type in ["electrical_plan", "network_diagram", "mechanical_drawing"]:
            if total_elements >= 50:
                complexity = "muy_alta"
                description = "Documento técnico de muy alta complejidad, requiere análisis especializado"
            elif total_elements >= 30:
                complexity = "alta"
                description = "Documento técnico complejo, típico de instalaciones industriales"
            elif total_elements >= 15:
                complexity = "moderada"
                description = "Documento técnico de complejidad media, aplicaciones comerciales"
            else:
                complexity = "basica"
                description = "Documento técnico básico, aplicaciones residenciales o simples"
        else:
            # Escalas ajustadas para otros tipos de documentos
            if total_elements >= 40:
                complexity = "muy_alta"
                description = "Documento muy complejo con múltiples elementos técnicos"
            elif total_elements >= 25:
                complexity = "alta"
                description = "Documento complejo con elementos técnicos avanzados"
            elif total_elements >= 12:
                complexity = "moderada"
                description = "Documento de complejidad moderada"
            else:
                complexity = "basica"
                description = "Documento técnico básico"
        
        return {
            "complexity_level": complexity,
            "total_elements": total_elements,
            "symbols_count": symbols_count,
            "routes_count": routes_count,
            "text_elements": text_elements,
            "description": description,
            "technical_domain": doc_type,
            "recommended_expertise": self._get_recommended_expertise(complexity, doc_type)
        }
    
    def _get_recommended_expertise(self, complexity: str, doc_type: str) -> str:
        """Recomienda el nivel de expertise necesario"""
        expertise_map = {
            ("muy_alta", "electrical_plan"): "Ingeniero eléctrico senior o especialista en alta tensión",
            ("alta", "electrical_plan"): "Ingeniero eléctrico o técnico especializado",
            ("moderada", "electrical_plan"): "Técnico eléctrico con experiencia",
            ("basica", "electrical_plan"): "Técnico eléctrico básico o electricista certificado",
            
            ("muy_alta", "network_diagram"): "Arquitecto de redes o ingeniero de telecomunicaciones",
            ("alta", "network_diagram"): "Administrador de redes senior",
            ("moderada", "network_diagram"): "Técnico en redes con experiencia",
            ("basica", "network_diagram"): "Técnico en sistemas básico",
            
            ("muy_alta", "process_diagram"): "Ingeniero de procesos o consultor industrial",
            ("alta", "process_diagram"): "Ingeniero industrial o de procesos",
            ("moderada", "process_diagram"): "Técnico en procesos con experiencia",
            ("basica", "process_diagram"): "Operador técnico capacitado"
        }
        
        return expertise_map.get((complexity, doc_type), "Profesional técnico con experiencia en el dominio")

