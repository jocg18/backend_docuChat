# Sistema de Análisis Técnico Universal

## Descripción General

Este sistema ha sido expandido para proporcionar capacidades avanzadas de análisis e interpretación de imágenes técnicas en general, no limitándose únicamente a planos eléctricos. Incluye reconocimiento de símbolos gráficos, análisis de rutas, interpretación de formas geométricas, análisis semántico de colores y generación de descripciones automáticas contextualizadas.

## Capacidades Expandidas

### 1. Tipos de Documentos Técnicos Soportados

- **Planos Eléctricos**: Circuitos, componentes eléctricos, sistemas de potencia
- **Esquemas de Postes y Redes**: Estructuras de soporte, redes de distribución eléctrica
- **Diagramas de Instalaciones**: Sistemas industriales, montajes técnicos
- **Mapas Técnicos**: Croquis de campo, ubicaciones técnicas
- **Diagramas de Procesos**: Flujos industriales, esquemas de proceso
- **Diagramas de Telecomunicaciones**: Topologías de red, sistemas de comunicación
- **Planos Arquitectónicos**: Estructuras, distribución espacial
- **Diagramas Mecánicos**: Equipos, maquinaria, componentes mecánicos
- **Esquemas de Control**: Automatización, instrumentación, control de procesos
- **Diagramas de Tuberías**: P&ID, sistemas de fluidos

### 2. Funcionalidades Principales

#### Detección Inteligente del Tipo de Documento
- Análisis automático basado en múltiples fuentes de evidencia
- Combinación de análisis de nombre de archivo, contenido visual, texto y estructura
- Confianza de clasificación con umbrales adaptativos

#### Reconocimiento de Símbolos Técnicos Universales
- Biblioteca expandida de símbolos por dominio técnico
- Detección combinada: YOLOv5 + análisis geométrico + reconocimiento de texto
- Clasificación contextual según el tipo de documento

#### Análisis de Rutas y Conexiones
- Detección de líneas, cables, tuberías, flujos
- Análisis de intersecciones y puntos de conexión
- Interpretación específica por dominio (eléctrico, hidráulico, datos, etc.)

#### Interpretación Semántica de Colores
- Mapeo de colores a significados técnicos por dominio
- Análisis de distribución cromática
- Interpretación contextual (códigos de colores técnicos)

#### Extracción y Clasificación de Textos
- OCR multiidioma (español/inglés)
- Clasificación de textos: especificaciones técnicas, etiquetas, dimensiones, notas
- Reconocimiento de patrones técnicos específicos

#### Generación de Descripciones Automáticas
- Descripciones contextualizadas por tipo de documento
- Resúmenes ejecutivos
- Evaluación de complejidad técnica

#### Respuestas a Preguntas Contextualizadas
- Sistema de Q&A inteligente sobre el contenido analizado
- Respuestas sobre cantidades, ubicaciones, funciones, colores
- Interpretación contextual basada en el análisis previo

## Estructura del Sistema

### Servicios Principales

1. **EnhancedUniversalAnalyzer**: Analizador principal con capacidades expandidas
2. **AdvancedImageAnalysisService**: Análisis avanzado específico para planos eléctricos
3. **UniversalTechnicalAnalyzer**: Analizador universal base
4. **ImageAnalysisService**: Servicio base para análisis de imágenes

### Nuevas Rutas de API

#### `/analyze-technical-universal` (POST)
Análisis técnico universal mejorado

**Parámetros:**
- `file`: Archivo de imagen (PNG, JPG, JPEG, BMP, TIFF, WEBP)
- `analysis_level`: Nivel de análisis (`basic`, `intermediate`, `advanced`, `expert`)
- `language`: Idioma de respuesta (`es`, `en`)

**Respuesta:**
```json
{
  "success": true,
  "message": "Análisis técnico universal completado",
  "results": {
    "analysis_id": "uuid",
    "document_type": {
      "type": "electrical_plan",
      "confidence": 0.87,
      "description": "Plano eléctrico con circuitos y componentes"
    },
    "symbols": [...],
    "routes": {...},
    "text_analysis": {...},
    "visual_analysis": {...},
    "technical_description": "...",
    "complexity_assessment": {...}
  }
}
```

#### `/ask-question` (POST)
Respuestas contextualizadas sobre análisis previos

**Parámetros:**
```json
{
  "question": "¿Cuántos símbolos eléctricos hay?",
  "analysis_results": {...},
  "language": "es"
}
```

**Respuesta:**
```json
{
  "success": true,
  "question": "¿Cuántos símbolos eléctricos hay?",
  "answer": "Se detectaron 12 símbolos técnicos en total. Desglose por tipo: - Interruptores: 4 - Tomacorrientes: 3 - Luminarias: 5",
  "language": "es"
}
```

#### `/analyze-document-type` (POST)
Detección únicamente del tipo de documento

#### `/analyze-electrical-advanced` (POST)
Análisis específico avanzado para planos eléctricos

#### `/get-analysis-info` (GET)
Información sobre capacidades del sistema

## Niveles de Análisis

### Basic
- Detección del tipo de documento
- Análisis básico de texto
- Descripción general

### Intermediate
- Todo lo anterior +
- Detección de símbolos técnicos
- Análisis básico de colores

### Advanced
- Todo lo anterior +
- Análisis completo de rutas y conexiones
- Análisis visual avanzado
- Interpretación semántica de colores

### Expert
- Todo lo anterior +
- Análisis de escalas y medidas
- Respuestas contextualizadas preparadas
- Análisis de complejidad técnica avanzado

## Casos de Uso

### 1. Análisis de Planos Eléctricos
```python
# Análisis específico para ingeniería eléctrica
results = analyzer.analyze_technical_image(
    "plano_electrico.jpg", 
    AnalysisComplexity.EXPERT
)

# Preguntas específicas
response = analyzer.answer_contextual_question(
    "¿Cuál es el voltaje de operación del sistema?",
    results
)
```

### 2. Análisis de Diagramas de Red
```python
# Detección automática de topología de red
results = analyzer.analyze_technical_image(
    "topologia_red.png",
    AnalysisComplexity.ADVANCED
)

# Consulta sobre conectividad
response = analyzer.answer_contextual_question(
    "¿Qué dispositivos de red están conectados?",
    results
)
```

### 3. Análisis de Procesos Industriales
```python
# Análisis de diagrama P&ID
results = analyzer.analyze_technical_image(
    "proceso_industrial.jpg",
    AnalysisComplexity.EXPERT
)

# Consulta sobre flujos
response = analyzer.answer_contextual_question(
    "¿Cuáles son las temperaturas de operación?",
    results
)
```

## Patrones de Reconocimiento

### Valores Eléctricos
- Voltajes: `240V`, `13.8kV`, `220 voltios`
- Corrientes: `15A`, `100 amperios`
- Potencias: `5kW`, `500 watts`
- Frecuencias: `60Hz`, `50 hertz`

### Valores de Red
- Velocidades: `1Gbps`, `100 Mbps`
- Direcciones IP: `192.168.1.1`
- Direcciones MAC: `00:1B:44:11:3A:B7`

### Valores de Proceso
- Temperaturas: `85°C`, `200 celsius`
- Presiones: `15 bar`, `200 psi`
- Flujos: `150 l/min`, `500 gpm`

### Dimensiones
- Métricas: `150mm`, `2.5m`, `10 cm`
- Imperiales: `6"`, `3 ft`, `2 yards`
- Diámetros: `ø25mm`

## Integración con Sistemas Existentes

### Para integrar con tu aplicación actual:

1. **Importar el nuevo servicio:**
```python
from app.services.enhanced_universal_analyzer import EnhancedUniversalAnalyzer
```

2. **Registrar las nuevas rutas:**
```python
from app.routes.enhanced_image import bp as enhanced_bp
app.register_blueprint(enhanced_bp, url_prefix='/api/enhanced')
```

3. **Usar el analizador:**
```python
analyzer = EnhancedUniversalAnalyzer(language="es")
results = analyzer.analyze_technical_image(image_path)
```

## Beneficios del Sistema Expandido

1. **Versatilidad**: Maneja múltiples tipos de documentos técnicos
2. **Precisión**: Detección automática del tipo con alta confianza
3. **Contextualización**: Interpretaciones específicas por dominio
4. **Interactividad**: Sistema de preguntas y respuestas
5. **Multiidioma**: Soporte en español e inglés
6. **Escalabilidad**: Niveles de análisis adaptativos
7. **Integración**: Compatible con sistemas existentes

## Consideraciones Técnicas

### Rendimiento
- Análisis básico: ~2-5 segundos
- Análisis avanzado: ~10-15 segundos
- Análisis experto: ~20-30 segundos

### Recursos
- RAM mínima: 4GB
- RAM recomendada: 8GB+
- GPU opcional para YOLO (mejora velocidad)

### Dependencias
- OpenCV 4.x
- PyTorch
- Tesseract OCR
- NumPy, PIL
- Flask para API

Este sistema expandido proporciona una solución integral para el análisis de imágenes técnicas, facilitando la interpretación automática y la generación de descripciones contextualizadas para una amplia variedad de documentos de ingeniería y técnicos.

