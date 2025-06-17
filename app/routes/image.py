from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from app.services.image_analysis_service import ImageAnalysisService

bp = Blueprint("image", __name__)

# Inicializar servicios
analyzer = ImageAnalysisService()  

# Inicializar analizador universal con manejo de errores
try:
    from app.services.enhanced_universal_analyzer import EnhancedUniversalAnalyzer, AnalysisComplexity
    enhanced_analyzer = EnhancedUniversalAnalyzer(language="es")  # nuevo analizador universal
except Exception as e:
    print(f"⚠️ Error inicializando EnhancedUniversalAnalyzer en image routes: {e}")
    enhanced_analyzer = None
    AnalysisComplexity = None

UPLOAD_FOLDER = "uploads/"
ALLOWED_IMG = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}


def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG


@bp.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    img = request.files["file"]
    if img.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_image(img.filename):
        return jsonify({"error": "Unsupported image type"}), 400

    filename = secure_filename(img.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img.save(img_path)

    result = analyzer.analyze_image(img_path)
    return jsonify({"message": "Imagen analizada", "result": result}), 200


@bp.route("/analyze-smart", methods=["POST"])
def analyze_smart():
    """Análisis inteligente que detecta automáticamente el tipo de imagen y aplica el análisis más apropiado"""
    global enhanced_analyzer
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó archivo"}), 400
        
        img_file = request.files["file"]
        if img_file.filename == "" or not allowed_image(img_file.filename):
            return jsonify({"error": "Archivo inválido"}), 400
        
        # Parámetros opcionales
        use_enhanced = request.form.get("enhanced", "true").lower() == "true"
        language = request.form.get("language", "es")
        analysis_level = request.form.get("analysis_level", "advanced")
        
        # Guardar archivo
        filename = secure_filename(img_file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)
        
        if use_enhanced and enhanced_analyzer is not None:
            # Usar analizador universal mejorado
            print(f"🚀 Análisis inteligente mejorado: {filename}")
            
            # Configurar idioma si es necesario
            if language != enhanced_analyzer.language:
                enhanced_analyzer = EnhancedUniversalAnalyzer(language=language)
            
            # Configurar nivel de análisis
            try:
                complexity_level = AnalysisComplexity(analysis_level)
            except ValueError:
                complexity_level = AnalysisComplexity.ADVANCED
            
            result = enhanced_analyzer.analyze_technical_image(img_path, complexity_level)
            
            # Limpiar archivo temporal
            try:
                os.remove(img_path)
            except:
                pass
            
            if "error" in result:
                return jsonify({
                    "success": False,
                    "error": result["error"],
                    "analyzer_used": "enhanced_universal"
                }), 500
            
            return jsonify({
                "success": True,
                "message": "Análisis inteligente completado con analizador universal",
                "analyzer_used": "enhanced_universal",
                "result": result
            }), 200
        
        else:
            # Usar analizador básico (compatible con versión anterior)
            fallback_reason = "enhanced_analyzer_not_available" if enhanced_analyzer is None else "basic_requested"
            print(f"⚡ Análisis básico: {filename} (razón: {fallback_reason})")
            result = analyzer.analyze_image(img_path)
            
            # Limpiar archivo temporal
            try:
                os.remove(img_path)
            except:
                pass
            
            return jsonify({
                "success": True,
                "message": "Análisis básico completado",
                "analyzer_used": "basic",
                "result": result
            }), 200
        
    except Exception as e:
        print(f"❌ Error en análisis inteligente: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}"
        }), 500


@bp.route("/analyze-capabilities", methods=["GET"])
def get_analyzer_capabilities():
    """Obtiene información sobre las capacidades disponibles"""
    try:
        capabilities = {
            "basic_analyzer": {
                "description": "Analizador básico de planos eléctricos",
                "capabilities": [
                    "Detección de símbolos eléctricos básicos",
                    "Análisis de rutas de cableado",
                    "OCR de textos",
                    "Detección de colores dominantes",
                    "Detección de formas geométricas"
                ],
                "supported_formats": ["png", "jpg", "jpeg"]
            },
            "enhanced_universal_analyzer": {
                "description": "Analizador universal mejorado para múltiples tipos de documentos técnicos",
                "capabilities": [
                    "Detección automática del tipo de documento",
                    "Reconocimiento de símbolos técnicos universales",
                    "Análisis avanzado de rutas y conexiones",
                    "Interpretación semántica de colores",
                    "OCR multiidioma",
                    "Generación de descripciones automáticas",
                    "Sistema de preguntas y respuestas contextualizadas",
                    "Análisis de escalas y proporciones",
                    "Múltiples niveles de análisis"
                ],
                "supported_formats": ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                "supported_languages": ["es", "en"],
                "analysis_levels": ["basic", "intermediate", "advanced", "expert"],
                "supported_document_types": [
                    "electrical_plan", "electrical_network", "pole_scheme",
                    "network_topology", "installation_diagram", "industrial_process",
                    "piping_diagram", "control_diagram", "mechanical_drawing",
                    "architectural_plan", "technical_map", "field_sketch"
                ]
            },
            "recommended_usage": {
                "basic_analyzer": "Para análisis rápido de planos eléctricos simples",
                "enhanced_universal_analyzer": "Para análisis completo de cualquier tipo de documento técnico"
            }
        }
        
        return jsonify({
            "success": True,
            "capabilities": capabilities
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error obteniendo capacidades: {str(e)}"
        }), 500
