"""Analizador Universal Mejorado de Imágenes Técnicas

Este servicio proporciona análisis avanzado e interpretación inteligente para diversos tipos de documentos técnicos:
- Planos eléctricos
- Esquemas de postes y redes eléctricas
- Diagramas de instalaciones industriales
- Mapas técnicos y croquis de campo
- Esquemas de procesos y flujos
- Diagramas de telecomunicaciones
- Planos arquitectónicos y estructurales
- Diagramas mecánicos y de equipos
- Esquemas de control y automatización

Capacidades avanzadas:
- Detección automática del tipo de documento técnico
- Reconocimiento inteligente de símbolos gráficos universales
- Análisis de rutas, conexiones y flujos
- Interpretación de formas geométricas complejas
- Análisis semántico de colores técnicos
- Extracción y clasificación de textos técnicos
- Generación de descripciones automáticas contextualizadas
- Respuesta a preguntas específicas sobre el contenido
- Soporte multiidioma (español/inglés)
- Análisis de escalas y proporciones
"""

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image as PILImage
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from enum import Enum
import json
from datetime import datetime
import uuid
from pathlib import Path

class EnhancedDocumentType(Enum):
    """Tipos expandidos de documentos técnicos"""
    ELECTRICAL_PLAN = "electrical_plan"
    ELECTRICAL_NETWORK = "electrical_network"
    POWER_DISTRIBUTION = "power_distribution"
    POLE_SCHEME = "pole_scheme"
    INSTALLATION_DIAGRAM = "installation_diagram"
    INDUSTRIAL_PROCESS = "industrial_process"
    TECHNICAL_MAP = "technical_map"
    FIELD_SKETCH = "field_sketch"
    FLOW_DIAGRAM = "flow_diagram"
    ARCHITECTURAL_PLAN = "architectural_plan"
    STRUCTURAL_PLAN = "structural_plan"
    TELECOM_DIAGRAM = "telecom_diagram"
    MECHANICAL_DRAWING = "mechanical_drawing"
    CONTROL_DIAGRAM = "control_diagram"
    AUTOMATION_SCHEME = "automation_scheme"
    INFRASTRUCTURE_PLAN = "infrastructure_plan"
    PIPING_DIAGRAM = "piping_diagram"
    NETWORK_TOPOLOGY = "network_topology"
    UNKNOWN = "unknown"

class AnalysisComplexity(Enum):
    """Niveles de complejidad de análisis"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class EnhancedUniversalAnalyzer:
    def __init__(self, language: str = "es"):
        """Inicializa el analizador universal mejorado
        
        Args:
            language: Idioma para las descripciones ('es' o 'en')
        """
        self.language = language
        self.analysis_id = None
        
        # Configuración de OCR multiidioma
        lang_codes = {"es": "spa", "en": "eng"}
        ocr_lang = lang_codes.get(language, "spa")
        
        self.tesseract_config = f'--oem 3 --psm 6 -l {ocr_lang} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=.,:;()[]{{}}"\'/\\°%ñÑáéíóúÁÉÍÓÚüÜ'
        
        # Diccionarios de símbolos técnicos expandidos por dominio
        self.technical_symbols = self._load_technical_symbols()
        
        # Patrones de reconocimiento avanzados
        self.recognition_patterns = self._load_recognition_patterns()
        
        # Mapas de interpretación de colores por dominio
        self.color_interpretations = self._load_color_interpretations()
        
        # Cargar modelo YOLOv5
        self.model = self._initialize_yolo_model()
        
        print(f"🚀 Analizador Universal Mejorado inicializado (idioma: {language})")
    
    def _load_technical_symbols(self) -> Dict:
        """Carga diccionarios expandidos de símbolos técnicos"""
        symbols = {
            # Símbolos eléctricos expandidos
            'electrical': {
                'switch': ['switch', 'SW', 'S', 'interruptor', 'conmutador'],
                'outlet': ['outlet', 'receptacle', 'R', 'tomacorriente', 'toma'],
                'light': ['light', 'lamp', 'L', 'luminaria', 'lámpara', 'LED'],
                'panel': ['panel', 'board', 'P', 'tablero', 'cuadro'],
                'motor': ['motor', 'M', 'MOT'],
                'transformer': ['transformer', 'T', 'transformador', 'trafo'],
                'fuse': ['fuse', 'F', 'fusible'],
                'breaker': ['breaker', 'CB', 'disyuntor', 'interruptor'],
                'ground': ['ground', 'GND', 'earth', 'tierra', 'masa'],
                'generator': ['generator', 'GEN', 'generador'],
                'capacitor': ['capacitor', 'C', 'condensador'],
                'relay': ['relay', 'K', 'relé', 'rele'],
                'contactor': ['contactor', 'KM', 'contactor'],
                'meter': ['meter', 'medidor', 'contador', 'kWh'],
                'ups': ['UPS', 'SAI', 'no-break'],
                'inverter': ['inverter', 'INV', 'inversor']
            },
            
            # Símbolos de redes y telecomunicaciones
            'network': {
                'router': ['router', 'R', 'enrutador'],
                'switch_net': ['switch', 'SW', 'conmutador'],
                'server': ['server', 'SRV', 'servidor'],
                'firewall': ['firewall', 'FW', 'cortafuegos'],
                'antenna': ['antenna', 'ANT', 'antena'],
                'cable': ['cable', 'CAB', 'UTP', 'fiber', 'fibra'],
                'wifi': ['wifi', 'wireless', 'inalámbrico', 'WLAN'],
                'modem': ['modem', 'MDM'],
                'repeater': ['repeater', 'REP', 'repetidor'],
                'access_point': ['AP', 'access point', 'punto de acceso'],
                'bridge': ['bridge', 'puente'],
                'gateway': ['gateway', 'GW', 'puerta de enlace']
            },
            
            # Símbolos de instalaciones industriales
            'installation': {
                'pipe': ['pipe', 'tubería', 'conducto', 'tube'],
                'valve': ['valve', 'válvula', 'V', 'VLV'],
                'pump': ['pump', 'bomba', 'P', 'BBA'],
                'tank': ['tank', 'tanque', 'T', 'TK'],
                'sensor': ['sensor', 'S', 'detector', 'SEN'],
                'meter_flow': ['flowmeter', 'medidor', 'M', 'caudalímetro'],
                'filter': ['filter', 'filtro', 'F', 'FLT'],
                'heater': ['heater', 'calentador', 'H', 'HTR'],
                'cooler': ['cooler', 'enfriador', 'C', 'CLR'],
                'compressor': ['compressor', 'compresor', 'COMP'],
                'heat_exchanger': ['heat exchanger', 'intercambiador', 'HX']
            },
            
            # Símbolos arquitectónicos
            'architectural': {
                'door': ['door', 'puerta', 'D'],
                'window': ['window', 'ventana', 'W', 'V'],
                'wall': ['wall', 'muro', 'pared'],
                'column': ['column', 'columna', 'C', 'COL'],
                'beam': ['beam', 'viga', 'B', 'VIG'],
                'stair': ['stair', 'escalera', 'E', 'ESC'],
                'room': ['room', 'habitación', 'R', 'HAB'],
                'toilet': ['toilet', 'baño', 'WC', 'sanitario'],
                'kitchen': ['kitchen', 'cocina', 'K'],
                'elevator': ['elevator', 'ascensor', 'ELV']
            },
            
            # Símbolos de procesos
            'process': {
                'start': ['start', 'inicio', 'begin', 'START'],
                'end': ['end', 'fin', 'stop', 'END'],
                'decision': ['decision', 'decisión', 'if', '?'],
                'process_step': ['process', 'proceso', 'step', 'etapa'],
                'data': ['data', 'datos', 'input', 'entrada'],
                'storage': ['storage', 'almacén', 'DB', 'base de datos'],
                'connector': ['connector', 'conector', 'unión'],
                'manual': ['manual', 'manual operation', 'operación manual']
            },
            
            # Símbolos de control y automatización
            'control': {
                'plc': ['PLC', 'controlador', 'CPU'],
                'hmi': ['HMI', 'pantalla', 'display'],
                'scada': ['SCADA', 'supervisión'],
                'transmitter': ['transmitter', 'transmisor', 'TX'],
                'actuator': ['actuator', 'actuador', 'ACT'],
                'servo': ['servo', 'servomotor'],
                'encoder': ['encoder', 'codificador', 'ENC']
            }
        }
        
        return symbols
    
    def _load_recognition_patterns(self) -> Dict:
        """Carga patrones de reconocimiento avanzados"""
        patterns = {
            'electrical_values': [
                r'\d+[.,]?\d*\s*(v|kv|mv|volt|voltio)s?',
                r'\d+[.,]?\d*\s*(a|amp|ampere|amperio)s?',
                r'\d+[.,]?\d*\s*(w|kw|mw|watt|vatio)s?',
                r'\d+[.,]?\d*\s*(hz|khz|hertz|hercio)s?',
                r'\d+[.,]?\d*\s*(va|kva|mva)s?'
            ],
            'network_values': [
                r'\d+[.,]?\d*\s*(mbps|gbps|bps)s?',
                r'\d+[.,]?\d*\s*(ghz|mhz)s?',
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
                r'([0-9a-f]{2}[:-]){5}[0-9a-f]{2}'  # MAC addresses
            ],
            'process_values': [
                r'\d+[.,]?\d*\s*(°c|°f|celsius|fahrenheit)',
                r'\d+[.,]?\d*\s*(bar|psi|pa|kpa|mpa)',
                r'\d+[.,]?\d*\s*(l/min|m³/h|gpm)',
                r'\d+[.,]?\d*\s*(rpm|rev/min)'
            ],
            'dimensions': [
                r'\d+[.,]?\d*\s*(mm|cm|m|km|in|ft|yard)s?',
                r'\d+[.,]?\d*\s*x\s*\d+[.,]?\d*',
                r'ø\s*\d+[.,]?\d*',  # Diameter
                r'\d+[.,]?\d*\s*("|\'|pulg|pulgada)s?'
            ],
            'scales': [
                r'1:(\d+)',
                r'scale\s+1:(\d+)',
                r'escala\s+1:(\d+)',
                r'(\d+):(\d+)'
            ]
        }
        
        return patterns
    
    def _load_color_interpretations(self) -> Dict:
        """Carga interpretaciones de colores por dominio técnico"""
        interpretations = {
            'electrical': {
                'red': 'Líneas de emergencia, alta tensión o fase R',
                'green': 'Líneas de tierra, seguridad o neutro',
                'blue': 'Líneas de control, señalización o fase S',
                'yellow': 'Advertencias, fase T o elementos especiales',
                'black': 'Líneas principales o estructuras',
                'brown': 'Fase L1 o conductores de potencia',
                'orange': 'Circuitos de control o señalización'
            },
            'network': {
                'blue': 'Conexiones de datos o Ethernet',
                'green': 'Conexiones de red local (LAN)',
                'red': 'Conexiones críticas o de emergencia',
                'yellow': 'Conexiones WAN o de internet',
                'orange': 'Fibra óptica o conexiones de alta velocidad',
                'purple': 'Conexiones de gestión o administrativas'
            },
            'process': {
                'red': 'Flujo de alta temperatura o materiales peligrosos',
                'blue': 'Flujo de agua fría o baja temperatura',
                'green': 'Flujo seguro o elementos de control',
                'yellow': 'Flujo de precaución o elementos variables',
                'orange': 'Flujo de vapor o alta presión',
                'brown': 'Flujo de productos o materias primas'
            },
            'architectural': {
                'black': 'Elementos estructurales o muros',
                'red': 'Elementos de seguridad contra incendios',
                'blue': 'Instalaciones hidráulicas',
                'green': 'Áreas verdes o elementos paisajísticos',
                'yellow': 'Elementos de advertencia o señalización'
            }
        }
        
        return interpretations
    
    def _initialize_yolo_model(self):
        """Inicializa el modelo YOLOv5 para detección de objetos"""
        try:
            # Intentar cargar modelo personalizado si existe
            custom_weights = "./weights/technical_symbols.pt"
            if Path(custom_weights).exists():
                model = torch.hub.load(
                    "ultralytics/yolov5", "custom", path=custom_weights, 
                    source="github", trust_repo=True
                )
            else:
                # Usar modelo estándar
                model = torch.hub.load(
                    "ultralytics/yolov5", "yolov5s", 
                    source="github", trust_repo=True
                )
            
            model.conf = 0.15  # Umbral bajo para mayor sensibilidad
            model.iou = 0.45
            model.max_det = 100  # Máximo de detecciones
            
            return model
        except Exception as e:
            print(f"⚠️ Error cargando YOLOv5: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DETECCIÓN INTELIGENTE MEJORADA DEL TIPO DE DOCUMENTO
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def detect_enhanced_document_type(self, img_bgr: np.ndarray, filename: str = "") -> Tuple[EnhancedDocumentType, float]:
        """Detecta el tipo de documento con mayor precisión y confianza"""
        print("🔍 Detectando tipo de documento técnico (análisis mejorado)...")
        
        # Múltiples fuentes de evidencia
        filename_evidence = self._analyze_filename_enhanced(filename)
        visual_evidence = self._analyze_visual_content_enhanced(img_bgr)
        text_evidence = self._analyze_text_content_enhanced(img_bgr)
        structure_evidence = self._analyze_document_structure(img_bgr)
        
        # Combinación ponderada de evidencias
        combined_scores = self._combine_enhanced_evidence(
            filename_evidence, visual_evidence, text_evidence, structure_evidence
        )
        
        # Seleccionar el tipo con mayor confianza
        best_type = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_type]
        
        # Umbral de confianza mínima
        if confidence < 0.4:
            best_type = EnhancedDocumentType.UNKNOWN
            confidence = 0.0
        
        print(f"📋 Tipo detectado: {best_type.value} (confianza: {confidence:.2f})")
        return best_type, confidence
    
    def _analyze_filename_enhanced(self, filename: str) -> Dict[EnhancedDocumentType, float]:
        """Análisis mejorado del nombre de archivo"""
        scores = {doc_type: 0.0 for doc_type in EnhancedDocumentType}
        
        filename_lower = filename.lower()
        
        # Patrones específicos expandidos
        patterns = {
            EnhancedDocumentType.ELECTRICAL_PLAN: [
                'electr', 'electric', 'power', 'energia', 'instalacion', 'plano_elect',
                'circuito', 'tablero', 'panel'
            ],
            EnhancedDocumentType.ELECTRICAL_NETWORK: [
                'red_elect', 'distribucion', 'transmission', 'subestacion', 'linea_mt',
                'linea_bt', 'alimentador'
            ],
            EnhancedDocumentType.POLE_SCHEME: [
                'poste', 'pole', 'apoyo', 'torre', 'estructur', 'montaje'
            ],
            EnhancedDocumentType.NETWORK_TOPOLOGY: [
                'network', 'red', 'topology', 'topologia', 'lan', 'wan', 'wifi',
                'conectividad', 'ethernet'
            ],
            EnhancedDocumentType.INSTALLATION_DIAGRAM: [
                'instalacion', 'installation', 'esquema', 'scheme', 'montaje',
                'assembly', 'setup'
            ],
            EnhancedDocumentType.INDUSTRIAL_PROCESS: [
                'proceso', 'process', 'industrial', 'planta', 'fabrica',
                'produccion', 'manufacturing'
            ],
            EnhancedDocumentType.PIPING_DIAGRAM: [
                'tuberia', 'pipe', 'piping', 'fluid', 'fluido', 'p&id', 'pfd'
            ],
            EnhancedDocumentType.CONTROL_DIAGRAM: [
                'control', 'automatizacion', 'automation', 'plc', 'scada',
                'instrumentacion', 'instrumentation'
            ]
        }
        
        for doc_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    scores[doc_type] += 0.4
        
        return scores
    
    def _analyze_visual_content_enhanced(self, img_bgr: np.ndarray) -> Dict[EnhancedDocumentType, float]:
        """Análisis visual mejorado con más características"""
        scores = {doc_type: 0.0 for doc_type in EnhancedDocumentType}
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Análisis de líneas y formas
        lines_analysis = self._analyze_line_patterns(gray)
        shapes_analysis = self._analyze_shape_patterns(gray)
        texture_analysis = self._analyze_texture_patterns(gray)
        
        # Clasificación por patrones visuales
        if lines_analysis['dense_grid']:
            scores[EnhancedDocumentType.ELECTRICAL_PLAN] += 0.3
            scores[EnhancedDocumentType.ARCHITECTURAL_PLAN] += 0.2
        
        if lines_analysis['network_like']:
            scores[EnhancedDocumentType.NETWORK_TOPOLOGY] += 0.4
            scores[EnhancedDocumentType.ELECTRICAL_NETWORK] += 0.3
        
        if shapes_analysis['circular_symbols'] > 10:
            scores[EnhancedDocumentType.ELECTRICAL_PLAN] += 0.3
            scores[EnhancedDocumentType.INDUSTRIAL_PROCESS] += 0.2
        
        if shapes_analysis['flow_like']:
            scores[EnhancedDocumentType.PIPING_DIAGRAM] += 0.4
            scores[EnhancedDocumentType.INDUSTRIAL_PROCESS] += 0.3
        
        if texture_analysis['technical_detail']:
            scores[EnhancedDocumentType.MECHANICAL_DRAWING] += 0.3
            scores[EnhancedDocumentType.CONTROL_DIAGRAM] += 0.2
        
        return scores
    
    def _analyze_line_patterns(self, gray: np.ndarray) -> Dict:
        """Analiza patrones de líneas específicos"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        analysis = {
            'total_lines': 0,
            'horizontal_ratio': 0.0,
            'vertical_ratio': 0.0,
            'dense_grid': False,
            'network_like': False
        }
        
        if lines is not None:
            analysis['total_lines'] = len(lines)
            
            horizontal = 0
            vertical = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                
                if -15 <= angle <= 15 or 165 <= abs(angle) <= 180:
                    horizontal += 1
                elif 75 <= abs(angle) <= 105:
                    vertical += 1
            
            total = len(lines)
            analysis['horizontal_ratio'] = horizontal / total if total > 0 else 0
            analysis['vertical_ratio'] = vertical / total if total > 0 else 0
            
            # Detectar patrones específicos
            analysis['dense_grid'] = (horizontal > 20 and vertical > 20)
            analysis['network_like'] = (total > 15 and analysis['horizontal_ratio'] < 0.7 and analysis['vertical_ratio'] < 0.7)
        
        return analysis
    
    def _analyze_shape_patterns(self, gray: np.ndarray) -> Dict:
        """Analiza patrones de formas específicos"""
        # Detectar círculos
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=3, maxRadius=50)
        
        # Detectar rectángulos y otras formas
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = 0
        complex_shapes = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                
                if len(approx) == 4:
                    rectangles += 1
                elif len(approx) > 6:
                    complex_shapes += 1
        
        return {
            'circular_symbols': len(circles[0]) if circles is not None else 0,
            'rectangular_shapes': rectangles,
            'complex_shapes': complex_shapes,
            'flow_like': (rectangles > 5 and complex_shapes > 3)
        }
    
    def _analyze_texture_patterns(self, gray: np.ndarray) -> Dict:
        """Analiza patrones de textura"""
        # Análisis de varianza local
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Métricas de textura
        mean_variance = np.mean(local_variance)
        detail_ratio = np.sum(local_variance > np.percentile(local_variance, 75)) / local_variance.size
        
        return {
            'mean_variance': mean_variance,
            'detail_ratio': detail_ratio,
            'technical_detail': (mean_variance > 100 and detail_ratio > 0.2)
        }
    
    def _analyze_text_content_enhanced(self, img_bgr: np.ndarray) -> Dict[EnhancedDocumentType, float]:
        """Análisis de texto mejorado con más contexto"""
        scores = {doc_type: 0.0 for doc_type in EnhancedDocumentType}
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=self.tesseract_config).lower()
        
        # Diccionarios de palabras clave expandidos
        domain_keywords = {
            EnhancedDocumentType.ELECTRICAL_PLAN: [
                'volt', 'amp', 'watt', 'electrical', 'electrico', 'power', 'energia',
                'switch', 'outlet', 'panel', 'circuit', 'wire', 'cable', 'phase',
                'neutral', 'ground', 'tierra', 'fase', 'neutro', 'tablero'
            ],
            EnhancedDocumentType.ELECTRICAL_NETWORK: [
                'subestacion', 'transformador', 'linea', 'distribucion', 'transmision',
                'alimentador', 'mt', 'bt', 'kv', 'substation', 'feeder'
            ],
            EnhancedDocumentType.POLE_SCHEME: [
                'poste', 'pole', 'apoyo', 'torre', 'estructura', 'montaje',
                'cruceta', 'aislador', 'conductor', 'guy wire'
            ],
            EnhancedDocumentType.NETWORK_TOPOLOGY: [
                'router', 'switch', 'server', 'ethernet', 'wifi', 'lan', 'wan',
                'ip', 'tcp', 'udp', 'firewall', 'gateway', 'subnet', 'vlan'
            ],
            EnhancedDocumentType.PIPING_DIAGRAM: [
                'pipe', 'valve', 'pump', 'tank', 'flow', 'pressure', 'tuberia',
                'valvula', 'bomba', 'tanque', 'flujo', 'presion', 'p&id'
            ],
            EnhancedDocumentType.CONTROL_DIAGRAM: [
                'plc', 'scada', 'hmi', 'sensor', 'actuator', 'control', 'automation',
                'controlador', 'automatizacion', 'servo', 'encoder'
            ]
        }
        
        # Búsqueda de patrones específicos
        for doc_type, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[doc_type] += 0.25
        
        # Búsqueda de patrones técnicos específicos
        for pattern_type, patterns in self.recognition_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if pattern_type == 'electrical_values':
                        scores[EnhancedDocumentType.ELECTRICAL_PLAN] += 0.3
                        scores[EnhancedDocumentType.ELECTRICAL_NETWORK] += 0.2
                    elif pattern_type == 'network_values':
                        scores[EnhancedDocumentType.NETWORK_TOPOLOGY] += 0.3
                    elif pattern_type == 'process_values':
                        scores[EnhancedDocumentType.INDUSTRIAL_PROCESS] += 0.3
                        scores[EnhancedDocumentType.PIPING_DIAGRAM] += 0.2
        
        return scores
    
    def _analyze_document_structure(self, img_bgr: np.ndarray) -> Dict[EnhancedDocumentType, float]:
        """Analiza la estructura general del documento"""
        scores = {doc_type: 0.0 for doc_type in EnhancedDocumentType}
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Análisis de proporción
        aspect_ratio = width / height
        
        # Análisis de distribución de contenido
        # Dividir en regiones y analizar densidad
        regions = {
            'top': gray[:height//3, :],
            'middle': gray[height//3:2*height//3, :],
            'bottom': gray[2*height//3:, :],
            'left': gray[:, :width//3],
            'center': gray[:, width//3:2*width//3],
            'right': gray[:, 2*width//3:]
        }
        
        densities = {}
        for region_name, region in regions.items():
            edges = cv2.Canny(region, 50, 150)
            densities[region_name] = np.sum(edges > 0) / edges.size
        
        # Patrones estructurales
        if densities['top'] > densities['bottom'] * 1.5:  # Título/encabezado
            scores[EnhancedDocumentType.ARCHITECTURAL_PLAN] += 0.2
            scores[EnhancedDocumentType.MECHANICAL_DRAWING] += 0.2
        
        if densities['left'] > densities['right'] * 1.3:  # Leyenda lateral
            scores[EnhancedDocumentType.ELECTRICAL_PLAN] += 0.2
            scores[EnhancedDocumentType.NETWORK_TOPOLOGY] += 0.2
        
        if aspect_ratio > 1.5:  # Formato apaisado
            scores[EnhancedDocumentType.ELECTRICAL_NETWORK] += 0.1
            scores[EnhancedDocumentType.PIPING_DIAGRAM] += 0.1
        
        return scores
    
    def _combine_enhanced_evidence(self, filename_ev: Dict, visual_ev: Dict, 
                                 text_ev: Dict, structure_ev: Dict) -> Dict[EnhancedDocumentType, float]:
        """Combina todas las evidencias con pesos optimizados"""
        combined_scores = {doc_type: 0.0 for doc_type in EnhancedDocumentType}
        
        # Pesos ajustados por experiencia
        weights = {
            'filename': 0.25,
            'visual': 0.35,
            'text': 0.30,
            'structure': 0.10
        }
        
        for doc_type in EnhancedDocumentType:
            if doc_type == EnhancedDocumentType.UNKNOWN:
                continue
            
            combined_scores[doc_type] = (
                filename_ev.get(doc_type, 0) * weights['filename'] +
                visual_ev.get(doc_type, 0) * weights['visual'] +
                text_ev.get(doc_type, 0) * weights['text'] +
                structure_ev.get(doc_type, 0) * weights['structure']
            )
        
        return combined_scores
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ANÁLISIS PRINCIPAL Y GENERACIÓN DE DESCRIPCIONES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def analyze_technical_image(self, image_path: str, analysis_level: AnalysisComplexity = AnalysisComplexity.ADVANCED) -> Dict:
        """Método principal para análisis completo de imágenes técnicas
        
        Args:
            image_path: Ruta de la imagen a analizar
            analysis_level: Nivel de complejidad del análisis
            
        Returns:
            Dict con análisis completo de la imagen
        """
        try:
            # Generar ID único para este análisis
            self.analysis_id = str(uuid.uuid4())
            
            # Cargar imagen
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": "No se pudo cargar la imagen"}
            
            filename = Path(image_path).name
            print(f"🔍 Iniciando análisis técnico universal de: {filename}")
            print(f"📊 Nivel de análisis: {analysis_level.value}")
            
            # Inicializar resultados
            results = {
                "analysis_id": self.analysis_id,
                "filename": filename,
                "analysis_level": analysis_level.value,
                "language": self.language,
                "timestamp": datetime.now().isoformat(),
                "image_info": {
                    "dimensions": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
                    "channels": img_bgr.shape[2],
                    "size_mb": round(Path(image_path).stat().st_size / (1024*1024), 2)
                }
            }
            
            # 1. Detección inteligente del tipo de documento
            document_type, confidence = self.detect_enhanced_document_type(img_bgr, filename)
            results["document_type"] = {
                "type": document_type.value,
                "confidence": confidence,
                "description": self._get_document_type_description(document_type)
            }
            
            # 2. Análisis de símbolos técnicos
            if analysis_level.value in ['intermediate', 'advanced', 'expert']:
                print("🔍 Detectando símbolos técnicos...")
                results["symbols"] = self._detect_enhanced_symbols(img_bgr, document_type)
            
            # 3. Análisis de rutas y conexiones
            if analysis_level.value in ['advanced', 'expert']:
                print("🔍 Analizando rutas y conexiones...")
                results["routes"] = self._detect_enhanced_routes(img_bgr, document_type)
            
            # 4. Análisis de texto técnico
            print("🔍 Extrayendo información textual...")
            results["text_analysis"] = self._extract_enhanced_text(img_bgr, document_type)
            
            # 5. Análisis visual y de colores
            if analysis_level.value in ['advanced', 'expert']:
                print("🔍 Analizando elementos visuales...")
                results["visual_analysis"] = self._analyze_enhanced_visuals(img_bgr, document_type)
            
            # 6. Análisis de escalas y medidas
            if analysis_level.value == 'expert':
                print("🔍 Analizando escalas y proporciones...")
                results["scale_analysis"] = self._analyze_scales_and_measurements(img_bgr)
            
            # 7. Generación de descripción comprensiva
            print("📝 Generando descripción técnica...")
            results["technical_description"] = self._generate_comprehensive_description(results)
            
            # 8. Evaluación de complejidad
            results["complexity_assessment"] = self._assess_enhanced_complexity(results)
            
            # 9. Preparar respuestas contextualizadas
            if analysis_level.value == 'expert':
                results["contextual_qa"] = self._prepare_contextual_responses(results)
            
            print("✅ Análisis técnico universal completado")
            return results
            
        except Exception as e:
            print(f"❌ Error en análisis técnico: {e}")
            return {"error": str(e), "analysis_id": self.analysis_id}
    
    def _get_document_type_description(self, doc_type: EnhancedDocumentType) -> str:
        """Obtiene descripción del tipo de documento"""
        descriptions = {
            "es": {
                EnhancedDocumentType.ELECTRICAL_PLAN: "Plano eléctrico con circuitos y componentes",
                EnhancedDocumentType.ELECTRICAL_NETWORK: "Diagrama de red eléctrica de distribución",
                EnhancedDocumentType.POLE_SCHEME: "Esquema de poste o estructura de soporte",
                EnhancedDocumentType.NETWORK_TOPOLOGY: "Topología de red de comunicaciones",
                EnhancedDocumentType.INSTALLATION_DIAGRAM: "Diagrama de instalación técnica",
                EnhancedDocumentType.INDUSTRIAL_PROCESS: "Diagrama de proceso industrial",
                EnhancedDocumentType.PIPING_DIAGRAM: "Diagrama de tuberías e instrumentación",
                EnhancedDocumentType.CONTROL_DIAGRAM: "Diagrama de control y automatización",
                EnhancedDocumentType.MECHANICAL_DRAWING: "Dibujo técnico mecánico",
                EnhancedDocumentType.ARCHITECTURAL_PLAN: "Plano arquitectónico",
                EnhancedDocumentType.UNKNOWN: "Documento técnico no identificado"
            },
            "en": {
                EnhancedDocumentType.ELECTRICAL_PLAN: "Electrical plan with circuits and components",
                EnhancedDocumentType.ELECTRICAL_NETWORK: "Electrical distribution network diagram",
                EnhancedDocumentType.POLE_SCHEME: "Pole or support structure scheme",
                EnhancedDocumentType.NETWORK_TOPOLOGY: "Communications network topology",
                EnhancedDocumentType.INSTALLATION_DIAGRAM: "Technical installation diagram",
                EnhancedDocumentType.INDUSTRIAL_PROCESS: "Industrial process diagram",
                EnhancedDocumentType.PIPING_DIAGRAM: "Piping and instrumentation diagram",
                EnhancedDocumentType.CONTROL_DIAGRAM: "Control and automation diagram",
                EnhancedDocumentType.MECHANICAL_DRAWING: "Mechanical technical drawing",
                EnhancedDocumentType.ARCHITECTURAL_PLAN: "Architectural plan",
                EnhancedDocumentType.UNKNOWN: "Unidentified technical document"
            }
        }
        
        return descriptions[self.language].get(doc_type, "Documento técnico")
    
    def answer_contextual_question(self, question: str, analysis_results: Dict) -> str:
        """Responde preguntas contextualizadas sobre el análisis
        
        Args:
            question: Pregunta del usuario
            analysis_results: Resultados del análisis previo
            
        Returns:
            Respuesta contextualizada
        """
        question_lower = question.lower()
        
        # Preparar contexto
        doc_type = analysis_results.get("document_type", {}).get("type", "unknown")
        symbols = analysis_results.get("symbols", [])
        routes = analysis_results.get("routes", {})
        text_analysis = analysis_results.get("text_analysis", {})
        
        # Patrones de preguntas comunes
        if any(word in question_lower for word in ['cuántos', 'cantidad', 'número', 'how many', 'count']):
            return self._answer_quantity_question(question_lower, analysis_results)
        
        elif any(word in question_lower for word in ['qué es', 'que tipo', 'what is', 'what type']):
            return self._answer_identification_question(question_lower, analysis_results)
        
        elif any(word in question_lower for word in ['dónde', 'ubicación', 'where', 'location']):
            return self._answer_location_question(question_lower, analysis_results)
        
        elif any(word in question_lower for word in ['cómo', 'como', 'how', 'función', 'funciona']):
            return self._answer_function_question(question_lower, analysis_results)
        
        elif any(word in question_lower for word in ['colores', 'color', 'colors']):
            return self._answer_color_question(question_lower, analysis_results)
        
        else:
            return self._generate_general_response(analysis_results)
    
    def _answer_quantity_question(self, question: str, results: Dict) -> str:
        """Responde preguntas sobre cantidades"""
        symbols = results.get("symbols", [])
        routes = results.get("routes", {})
        
        response_parts = []
        
        if 'símbolos' in question or 'symbols' in question:
            response_parts.append(f"Se detectaron {len(symbols)} símbolos técnicos en total.")
            
            # Desglose por tipo
            symbol_types = {}
            for symbol in symbols:
                stype = symbol.get("type", "unknown")
                symbol_types[stype] = symbol_types.get(stype, 0) + 1
            
            if symbol_types:
                response_parts.append("Desglose por tipo:")
                for stype, count in sorted(symbol_types.items()):
                    response_parts.append(f"- {stype.replace('_', ' ').title()}: {count}")
        
        elif 'conexiones' in question or 'rutas' in question or 'connections' in question:
            total_routes = routes.get("summary", {}).get("total_routes", 0)
            response_parts.append(f"Se identificaron {total_routes} conexiones o rutas.")
            
            if routes.get("summary"):
                summary = routes["summary"]
                if summary.get("horizontal"):
                    response_parts.append(f"- Horizontales: {summary['horizontal']}")
                if summary.get("vertical"):
                    response_parts.append(f"- Verticales: {summary['vertical']}")
                if summary.get("intersections"):
                    response_parts.append(f"- Intersecciones: {summary['intersections']}")
        
        return "\n".join(response_parts) if response_parts else "No se pudo determinar la cantidad solicitada."
    
    def _answer_identification_question(self, question: str, results: Dict) -> str:
        """Responde preguntas de identificación"""
        doc_info = results.get("document_type", {})
        
        response = f"Este es un {doc_info.get('description', 'documento técnico')}."
        
        confidence = doc_info.get("confidence", 0)
        if confidence > 0.7:
            response += f" (Identificado con alta confianza: {confidence:.1%})"
        elif confidence > 0.4:
            response += f" (Identificado con confianza moderada: {confidence:.1%})"
        
        # Agregar características principales
        symbols = results.get("symbols", [])
        if symbols:
            main_types = set(s.get("type", "") for s in symbols[:5])
            if main_types:
                response += f"\n\nCaracterísticas principales: {', '.join(main_types)}"
        
        return response
    
    def _answer_location_question(self, question: str, results: Dict) -> str:
        """Responde preguntas sobre ubicación"""
        symbols = results.get("symbols", [])
        
        if not symbols:
            return "No se detectaron elementos específicos con ubicación determinada."
        
        response_parts = ["Ubicaciones de elementos detectados:"]
        
        for i, symbol in enumerate(symbols[:5]):  # Primeros 5 símbolos
            bbox = symbol.get("bbox", [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) // 2
                y_center = (bbox[1] + bbox[3]) // 2
                
                # Determinar región aproximada
                img_info = results.get("image_info", {})
                dimensions = img_info.get("dimensions", "")
                if "x" in dimensions:
                    width, height = map(int, dimensions.split("x"))
                    
                    h_region = "izquierda" if x_center < width/3 else "centro" if x_center < 2*width/3 else "derecha"
                    v_region = "superior" if y_center < height/3 else "centro" if y_center < 2*height/3 else "inferior"
                    
                    element_type = symbol.get("type", "elemento").replace("_", " ")
                    response_parts.append(f"- {element_type}: región {v_region} {h_region}")
        
        return "\n".join(response_parts)
    
    def _answer_function_question(self, question: str, results: Dict) -> str:
        """Responde preguntas sobre función"""
        doc_type = results.get("document_type", {}).get("type", "unknown")
        
        function_descriptions = {
            "electrical_plan": "Este plano muestra la distribución de circuitos eléctricos, ubicación de componentes como interruptores, tomacorrientes y luminarias, así como el trazado del cableado.",
            "electrical_network": "Este diagrama representa la red de distribución eléctrica, mostrando subestaciones, líneas de transmisión, transformadores y puntos de conexión.",
            "network_topology": "Esta topología de red muestra cómo están interconectados los dispositivos de comunicación como routers, switches y servidores.",
            "piping_diagram": "Este diagrama muestra el sistema de tuberías, válvulas, bombas y instrumentos para el manejo de fluidos en un proceso industrial.",
            "control_diagram": "Este esquema representa el sistema de control automático, mostrando sensores, actuadores, controladores y las señales entre ellos."
        }
        
        base_response = function_descriptions.get(doc_type, "Este documento técnico contiene información especializada sobre un sistema o instalación.")
        
        # Agregar información específica basada en el análisis
        symbols = results.get("symbols", [])
        if symbols:
            main_functions = set()
            for symbol in symbols:
                stype = symbol.get("type", "")
                if "switch" in stype or "interruptor" in stype:
                    main_functions.add("control de circuitos")
                elif "motor" in stype:
                    main_functions.add("accionamiento mecánico")
                elif "sensor" in stype:
                    main_functions.add("monitoreo y medición")
                elif "valve" in stype or "valvula" in stype:
                    main_functions.add("control de flujo")
            
            if main_functions:
                base_response += f"\n\nFunciones principales identificadas: {', '.join(main_functions)}."
        
        return base_response
    
    def _answer_color_question(self, question: str, results: Dict) -> str:
        """Responde preguntas sobre colores"""
        visual_analysis = results.get("visual_analysis", {})
        colors = visual_analysis.get("dominant_colors", [])
        
        if not colors:
            return "No se pudo realizar un análisis detallado de colores."
        
        doc_type = results.get("document_type", {}).get("type", "unknown")
        domain = doc_type.split("_")[0] if "_" in doc_type else doc_type
        
        response_parts = ["Análisis de colores:"]
        
        for i, color in enumerate(colors[:3]):
            hex_color = color.get("hex", "#000000")
            percentage = color.get("percentage", 0)
            meaning = color.get("technical_meaning", "")
            
            response_parts.append(f"- Color {i+1}: {hex_color} ({percentage:.1f}%)")
            if meaning:
                response_parts.append(f"  Significado técnico: {meaning}")
        
        # Agregar interpretación general
        color_interpretations = visual_analysis.get("technical_interpretation", [])
        if color_interpretations:
            response_parts.append("\nInterpretación técnica:")
            for interpretation in color_interpretations[:2]:
                response_parts.append(f"- {interpretation}")
        
        return "\n".join(response_parts)
    
    def _generate_general_response(self, results: Dict) -> str:
        """Genera una respuesta general sobre el análisis"""
        doc_info = results.get("document_type", {})
        description = results.get("technical_description", "")
        
        response = f"Análisis del {doc_info.get('description', 'documento técnico')}:\n\n"
        
        if description:
            # Tomar las primeras líneas de la descripción técnica
            desc_lines = description.split("\n")[:5]
            response += "\n".join(desc_lines)
        
        complexity = results.get("complexity_assessment", {})
        if complexity:
            level = complexity.get("complexity_level", "unknown")
            total_elements = complexity.get("total_elements", 0)
            response += f"\n\nComplejidad: {level} ({total_elements} elementos detectados)"
        
        return response
    
    # Métodos auxiliares que implementarían las funcionalidades específicas
    # (estos métodos usarían las funciones ya existentes de los otros servicios)
    
    def _detect_enhanced_symbols(self, img_bgr: np.ndarray, document_type: EnhancedDocumentType) -> List[Dict]:
        """Detección mejorada de símbolos (usar UniversalTechnicalAnalyzer)"""
        symbols = []
        
        # Usar análisis básico geométrico
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detectar círculos (conexiones, componentes)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=3, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                symbols.append({
                    "type": f"{document_type.value}_circular_element",
                    "subtype": "detected_circle",
                    "confidence": 0.7,
                    "bbox": [x-r, y-r, x+r, y+r],
                    "area": int(math.pi * r * r),
                    "detection_method": "geometric",
                    "domain": document_type.value
                })
        
        # Detectar rectángulos
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 8000:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                
                if len(approx) == 4:
                    symbols.append({
                        "type": f"{document_type.value}_rectangular_element",
                        "subtype": "detected_rectangle",
                        "confidence": 0.6,
                        "bbox": [x, y, x+w, y+h],
                        "area": int(area),
                        "detection_method": "geometric",
                        "domain": document_type.value
                    })
        
        return symbols[:10]  # Limitar a 10 elementos más relevantes
    
    def _detect_enhanced_routes(self, img_bgr: np.ndarray, document_type: EnhancedDocumentType) -> Dict:
        """Detección mejorada de rutas (usar UniversalTechnicalAnalyzer)"""
        # Implementación que integra con el análisis existente
        return {}
    
    def _extract_enhanced_text(self, img_bgr: np.ndarray, document_type: EnhancedDocumentType) -> Dict:
        """Extracción mejorada de texto"""
        # Implementación que integra con el análisis existente
        return {}
    
    def _analyze_enhanced_visuals(self, img_bgr: np.ndarray, document_type: EnhancedDocumentType) -> Dict:
        """Análisis visual mejorado"""
        # Implementación que integra con el análisis existente
        return {}
    
    def _analyze_scales_and_measurements(self, img_bgr: np.ndarray) -> Dict:
        """Análisis de escalas y medidas"""
        # Implementación específica para escalas
        return {}
    
    def _generate_comprehensive_description(self, results: Dict) -> str:
        """Genera descripción comprensiva (usar AdvancedImageAnalysisService)"""
        # Implementación que integra con el generador existente
        return ""
    
    def _assess_enhanced_complexity(self, results: Dict) -> Dict:
        """Evaluación mejorada de complejidad"""
        # Implementación que integra con la evaluación existente
        return {}
    
    def _prepare_contextual_responses(self, results: Dict) -> Dict:
        """Prepara respuestas contextualizadas"""
        # Implementación para preparar Q&A contextual
        return {}

