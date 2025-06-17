from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename

bp = Blueprint("enhanced_image", __name__)

# Inicializar servicios de análisis con manejo de errores
try:
    from app.services.enhanced_universal_analyzer import EnhancedUniversalAnalyzer, AnalysisComplexity
    enhanced_analyzer = EnhancedUniversalAnalyzer(language="es")
except Exception as e:
    print(f"⚠️ Error inicializando EnhancedUniversalAnalyzer: {e}")
    enhanced_analyzer = None
    
try:
    from app.services.advanced_image_analysis import AdvancedImageAnalysisService
    advanced_analyzer = AdvancedImageAnalysisService()
except Exception as e:
    print(f"⚠️ Error inicializando AdvancedImageAnalysisService: {e}")
    advanced_analyzer = None
    
try:
    from app.services.universal_technical_analyzer import UniversalTechnicalAnalyzer
    universal_analyzer = UniversalTechnicalAnalyzer()
except Exception as e:
    print(f"⚠️ Error inicializando UniversalTechnicalAnalyzer: {e}")
    universal_analyzer = None

UPLOAD_FOLDER = "uploads/"
ALLOWED_IMG = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG

@bp.route("/analyze-technical-universal", methods=["POST"])
def analyze_technical_universal():
    """Análisis técnico universal mejorado de imágenes"""
    global enhanced_analyzer
    try:
        # Validar archivo
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó archivo"}), 400
        
        img_file = request.files["file"]
        if img_file.filename == "":
            return jsonify({"error": "No se seleccionó archivo"}), 400
        
        if not allowed_image(img_file.filename):
            return jsonify({"error": "Tipo de imagen no soportado"}), 400
        
        # Obtener parámetros opcionales
        analysis_level = request.form.get("analysis_level", "advanced")
        language = request.form.get("language", "es")
        
        # Validar nivel de análisis
        try:
            complexity_level = AnalysisComplexity(analysis_level)
        except ValueError:
            complexity_level = AnalysisComplexity.ADVANCED
        
        # Guardar archivo
        filename = secure_filename(img_file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)
        
        # Configurar analizador según idioma
        if language != enhanced_analyzer.language:
            enhanced_analyzer = EnhancedUniversalAnalyzer(language=language)
        
        # Validar que el analizador esté disponible
        if enhanced_analyzer is None:
            return jsonify({
                "success": False,
                "error": "Analizador universal no disponible. Revise la configuración."
            }), 503
        
        # Realizar análisis técnico universal
        print(f"🔍 Iniciando análisis técnico universal: {filename}")
        results = enhanced_analyzer.analyze_technical_image(img_path, complexity_level)
        
        # Limpiar archivo temporal (opcional)
        try:
            os.remove(img_path)
        except:
            pass
        
        if "error" in results:
            return jsonify({
                "success": False,
                "error": results["error"],
                "analysis_id": results.get("analysis_id")
            }), 500
        
        return jsonify({
            "success": True,
            "message": "Análisis técnico universal completado",
            "results": results
        }), 200
        
    except Exception as e:
        print(f"❌ Error en análisis técnico universal: {e}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}"
        }), 500

@bp.route("/analyze-electrical-advanced", methods=["POST"])
def analyze_electrical_advanced():
    """Análisis avanzado específico para planos eléctricos"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó archivo"}), 400
        
        img_file = request.files["file"]
        if img_file.filename == "" or not allowed_image(img_file.filename):
            return jsonify({"error": "Archivo inválido"}), 400
        
        # Guardar archivo
        filename = secure_filename(img_file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)
        
        # Análisis específico para planos eléctricos
        print(f"⚡ Analizando plano eléctrico: {filename}")
        results = advanced_analyzer.analyze_electrical_plan(img_path)
        
        # Limpiar archivo temporal
        try:
            os.remove(img_path)
        except:
            pass
        
        if "error" in results:
            return jsonify({
                "success": False,
                "error": results["error"]
            }), 500
        
        return jsonify({
            "success": True,
            "message": "Análisis avanzado de plano eléctrico completado",
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error en análisis eléctrico: {str(e)}"
        }), 500

@bp.route("/ask-question", methods=["POST"])
def ask_contextual_question():
    """Responde preguntas contextualizadas sobre un análisis previo"""
    global enhanced_analyzer
    try:
        data = request.get_json()
        
        if not data or "question" not in data or "analysis_results" not in data:
            return jsonify({"error": "Datos incompletos. Se requiere 'question' y 'analysis_results'"}), 400
        
        question = data["question"]
        analysis_results = data["analysis_results"]
        language = data.get("language", "es")
        
        # Configurar analizador según idioma
        if language != enhanced_analyzer.language:
            enhanced_analyzer = EnhancedUniversalAnalyzer(language=language)
        
        # Generar respuesta contextualizada
        print(f"❓ Respondiendo pregunta: {question[:50]}...")
        response = enhanced_analyzer.answer_contextual_question(question, analysis_results)
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": response,
            "language": language
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error respondiendo pregunta: {str(e)}"
        }), 500

@bp.route("/analyze-document-type", methods=["POST"])
def analyze_document_type_only():
    """Detecta únicamente el tipo de documento técnico"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó archivo"}), 400
        
        img_file = request.files["file"]
        if img_file.filename == "" or not allowed_image(img_file.filename):
            return jsonify({"error": "Archivo inválido"}), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(img_file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)
        
        # Solo detectar tipo de documento
        import cv2
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return jsonify({"error": "No se pudo procesar la imagen"}), 400
        
        document_type, confidence = enhanced_analyzer.detect_enhanced_document_type(img_bgr, filename)
        description = enhanced_analyzer._get_document_type_description(document_type)
        
        # Limpiar archivo temporal
        try:
            os.remove(img_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "document_type": {
                "type": document_type.value,
                "confidence": confidence,
                "description": description
            },
            "filename": filename
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error detectando tipo: {str(e)}"
        }), 500

@bp.route("/get-analysis-info", methods=["GET"])
def get_analysis_info():
    """Obtiene información sobre las capacidades del analizador"""
    try:
        info = {
            "analyzer_version": "EnhancedUniversalAnalyzer v1.0",
            "supported_languages": ["es", "en"],
            "analysis_levels": [level.value for level in AnalysisComplexity],
            "supported_document_types": [
                {
                    "type": doc_type.value,
                    "description_es": enhanced_analyzer._get_document_type_description(doc_type) if enhanced_analyzer.language == "es" else None,
                    "description_en": EnhancedUniversalAnalyzer("en")._get_document_type_description(doc_type)
                }
                for doc_type in enhanced_analyzer.technical_symbols.keys() if hasattr(doc_type, 'value')
            ],
            "capabilities": [
                "Detección automática del tipo de documento",
                "Reconocimiento de símbolos técnicos universales",
                "Análisis de rutas y conexiones",
                "Interpretación de colores técnicos",
                "Extracción y clasificación de textos",
                "Generación de descripciones automáticas",
                "Respuestas contextualizadas",
                "Análisis de escalas y proporciones",
                "Soporte multiidioma"
            ],
            "supported_formats": list(ALLOWED_IMG)
        }
        
        return jsonify({
            "success": True,
            "info": info
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error obteniendo información: {str(e)}"
        }), 500

