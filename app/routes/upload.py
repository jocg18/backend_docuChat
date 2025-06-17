from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid

from app.services.document_processor import DocumentProcessor
from app.services.vector_database import VectorDatabaseService

bp = Blueprint('upload', __name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # 1) Guardar el archivo
    filename = secure_filename(file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 2) Procesar el documento con análisis avanzado
    processor = DocumentProcessor()
    try:
        # Generar namespace único para este archivo
        file_namespace = str(uuid.uuid4())
        
        # Procesar con análisis universal habilitado (modo automático)
        content, vectors, processing_result = processor.process_file(
            filepath, 
            namespace=file_namespace, 
            use_universal_analysis=True,
            analysis_mode="auto"
        )
        
        print(f"🎯 Archivo procesado con análisis universal:")
        print(f"   - Nombre: {processing_result['filename']}")
        print(f"   - File ID: {processing_result['file_id']}")
        print(f"   - Namespace: {processing_result['namespace']}")
        print(f"   - Modo de análisis: {processing_result['analysis_mode']}")
        print(f"   - Embeddings generados: {processing_result['chunks_created']}")
        
        # Determinar el tipo de análisis realizado
        has_universal = processing_result.get('universal_analysis') is not None
        has_advanced = processing_result.get('advanced_analysis') is not None
        
        analysis_type = "basic_image"
        if has_universal:
            doc_type = processing_result['universal_analysis'].get('document_type', 'unknown')
            analysis_type = f"universal_{doc_type}"
        elif has_advanced:
            analysis_type = "advanced_electrical_plan"
        
    except Exception as e:
        print("❌ Error al procesar el documento:", str(e))
        return jsonify({"error": "Error al procesar el documento", "details": str(e)}), 500

    # 3) Preparar respuesta con información completa del análisis
    response_data = {
        "message": f"Archivo '{filename}' procesado exitosamente",
        "file_info": {
            "filename": processing_result['filename'],
            "file_id": processing_result['file_id'],
            "namespace": processing_result['namespace'],
            "content_length": processing_result['content_length'],
            "embeddings_created": processing_result['chunks_created']
        },
        "analysis_type": analysis_type,
        "processing_timestamp": processing_result['processing_timestamp']
    }
    
    # 4) Incluir análisis básico si está disponible
    if processing_result.get('basic_analysis'):
        basic = processing_result['basic_analysis']
        response_data['basic_analysis'] = {
            "symbols_detected": len(basic.get('symbols', [])),
            "routes_detected": len(basic.get('routes', [])),
            "text_elements": len(basic.get('texts', [])),
            "dominant_colors": len(basic.get('colors', [])),
            "shapes_detected": len(basic.get('shapes', []))
        }
    
    # 5) Incluir análisis avanzado si está disponible
    if processing_result.get('advanced_analysis'):
        advanced = processing_result['advanced_analysis']
        response_data['advanced_analysis'] = {
            "executive_summary": advanced.get('descriptions', {}).get('executive_summary', ''),
            "complexity_assessment": advanced.get('summary', {}),
            "components_summary": advanced.get('components_detected', {}),
            "key_technical_specs": {
                "electrical_labels": len(advanced.get('technical_specifications', {}).get('electrical_labels', [])),
                "dimensions_found": len(advanced.get('technical_specifications', {}).get('dimensions', [])),
                "color_coding_detected": len(advanced.get('technical_specifications', {}).get('color_coding', []))
            }
        }
        
        # Incluir descripción comprensiva si es muy detallada
        comprehensive_desc = advanced.get('descriptions', {}).get('comprehensive', '')
        if len(comprehensive_desc) > 100:
            response_data['advanced_analysis']['comprehensive_description'] = comprehensive_desc[:500] + "..."
    
    # 6) Incluir análisis universal si está disponible
    if processing_result.get('universal_analysis'):
        universal = processing_result['universal_analysis']
        response_data['universal_analysis'] = {
            "document_type": universal.get('document_type', 'unknown'),
            "executive_summary": universal.get('descriptions', {}).get('executive_summary', ''),
            "complexity_assessment": universal.get('summary', {}),
            "components_summary": universal.get('components_detected', {}),
            "key_technical_specs": {
                "technical_specifications": len(universal.get('technical_specifications', {}).get('technical_specifications', [])),
                "labels_found": len(universal.get('technical_specifications', {}).get('labels', [])),
                "dimensions_found": len(universal.get('technical_specifications', {}).get('dimensions', [])),
                "visual_interpretations": len(universal.get('technical_specifications', {}).get('visual_interpretation', []))
            }
        }
        
        # Incluir descripción comprensiva del análisis universal
        comprehensive_desc = universal.get('descriptions', {}).get('comprehensive', '')
        if len(comprehensive_desc) > 100:
            response_data['universal_analysis']['comprehensive_description'] = comprehensive_desc[:500] + "..."
    
    print(f"✅ Respuesta preparada para archivo: {filename}")
    return jsonify(response_data), 200
