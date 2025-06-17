# backend/app/routes/query.py
from flask import Blueprint, request, jsonify
from app.services.chatbot_service import ChatbotService
from app.services.enhanced_query_service import EnhancedQueryService

bp = Blueprint('query', __name__)
chatbot = ChatbotService()
enhanced_query_service = EnhancedQueryService()

@bp.route('/query', methods=['POST'])
def handle_query():
    """Maneja consultas con capacidades mejoradas y vinculación por archivo"""
    data = request.get_json()
    question = data.get('query')
    
    # Nuevos parámetros opcionales
    namespace = data.get('namespace')  # Para consultas específicas a un archivo
    file_id = data.get('file_id')  # Compatibilidad con el sistema anterior
    query_type = data.get('query_type', 'enhanced')  # 'enhanced', 'legacy', 'intelligent'
    content_filter = data.get('content_filter')  # 'electrical_plan', 'general_image'
    complexity_filter = data.get('complexity_filter')  # 'basica', 'moderada', 'alta', 'muy_alta'
    
    if not question:
        return jsonify({"error": "No query provided"}), 400
    
    print(f"💬 Nueva consulta recibida:")
    print(f"   - Pregunta: {question}")
    print(f"   - Tipo: {query_type}")
    print(f"   - Namespace: {namespace}")
    print(f"   - Filtros: content={content_filter}, complexity={complexity_filter}")
    
    try:
        # Seleccionar tipo de consulta
        # Usar file_id como namespace si no hay namespace específico
        target_namespace = namespace or file_id
        
        if query_type == 'enhanced' and target_namespace:
            # Consulta específica a un archivo
            print(f"🎯 Usando namespace específico: {target_namespace}")
            result = enhanced_query_service.query_specific_file(
                query=question,
                namespace=target_namespace,
                top_k=5
            )
            
            if result["success"]:
                return jsonify({
                    "type": "file_specific_query",
                    "query": question,
                    "namespace": target_namespace,
                    "answer": result["response"],
                    "confidence": result["confidence"],
                    "file_context": result["file_context"],
                    "relevant_chunks": result["relevant_chunks"],
                    "metadata": {
                        "analysis_type": "enhanced",
                        "results_count": result["results_count"]
                    }
                }), 200
            else:
                # Fallback: Intentar búsqueda general si la específica falla
                print(f"⚠️ No se encontraron resultados en namespace {target_namespace}, intentando búsqueda general")
                fallback_result = enhanced_query_service.intelligent_search(
                    query=question,
                    top_k=5
                )
                
                if fallback_result["success"]:
                    return jsonify({
                        "type": "file_specific_query_with_fallback",
                        "query": question,
                        "original_namespace": target_namespace,
                        "answer": fallback_result["intelligent_response"],
                        "files_found": fallback_result["files_found"],
                        "total_matches": fallback_result["total_matches"],
                        "file_summaries": fallback_result["file_summaries"],
                        "metadata": {
                            "analysis_type": "enhanced_fallback",
                            "original_error": result.get("message", "Error en consulta específica"),
                            "fallback_used": True
                        },
                        "warning": f"No se encontraron resultados en el archivo específico. Se realizó búsqueda general."
                    }), 200
                else:
                    return jsonify({
                        "type": "file_specific_query",
                        "error": "No se encontraron resultados ni en el archivo específico ni en búsqueda general",
                        "namespace": target_namespace,
                        "query": question,
                        "details": {
                            "specific_error": result.get("message", "Error en consulta específica"),
                            "fallback_error": fallback_result.get("message", "Error en búsqueda general")
                        },
                        "suggestion": "Verifica que el archivo esté correctamente cargado o intenta reformular la pregunta"
                    }), 404
        
        elif query_type == 'intelligent':
            # Búsqueda inteligente con filtros
            result = enhanced_query_service.intelligent_search(
                query=question,
                content_filter=content_filter,
                complexity_filter=complexity_filter,
                top_k=10
            )
            
            if result["success"]:
                return jsonify({
                    "type": "intelligent_search",
                    "query": question,
                    "answer": result["intelligent_response"],
                    "filters_applied": result["filters_applied"],
                    "files_found": result["files_found"],
                    "total_matches": result["total_matches"],
                    "file_summaries": result["file_summaries"],
                    "metadata": {
                        "analysis_type": "intelligent_filtered"
                    }
                }), 200
            else:
                return jsonify({
                    "type": "intelligent_search",
                    "error": result.get("message", "No se encontraron resultados"),
                    "filters_applied": result.get("filters_applied", {}),
                    "query": question
                }), 404
        
        elif query_type == 'legacy' and (file_id or namespace):
            # Sistema de consulta original (compatibilidad)
            target_namespace = namespace or file_id
            if not target_namespace:
                return jsonify({"error": "No file_id or namespace provided for legacy query"}), 400
            
            answer = chatbot.answer_query(question, target_namespace)
            return jsonify({
                "type": "legacy_query",
                "query": question,
                "file_id": target_namespace,
                "answer": answer,
                "metadata": {
                    "analysis_type": "legacy"
                }
            }), 200
        
        else:
            # Búsqueda general mejorada (por defecto)
            result = enhanced_query_service.intelligent_search(
                query=question,
                top_k=8
            )
            
            if result["success"]:
                return jsonify({
                    "type": "general_enhanced_search",
                    "query": question,
                    "answer": result["intelligent_response"],
                    "files_found": result["files_found"],
                    "total_matches": result["total_matches"],
                    "file_summaries": result["file_summaries"][:5],  # Top 5 archivos
                    "metadata": {
                        "analysis_type": "enhanced_general",
                        "suggestion": "Para mejores resultados, especifica un namespace o usa filtros"
                    }
                }), 200
            else:
                return jsonify({
                    "type": "general_enhanced_search",
                    "error": "No se encontraron resultados relevantes",
                    "query": question,
                    "suggestion": "Intenta reformular tu pregunta o verifica que los archivos estén cargados"
                }), 404
    
    except Exception as e:
        print(f"❌ Error procesando consulta: {e}")
        return jsonify({
            "error": "Error interno procesando la consulta",
            "details": str(e),
            "query": question,
            "query_type": query_type
        }), 500

@bp.route('/query/multiple', methods=['POST'])
def handle_multiple_files_query():
    """Maneja consultas en múltiples archivos específicos"""
    data = request.get_json()
    question = data.get('query')
    namespaces = data.get('namespaces', [])
    top_k = data.get('top_k', 3)
    
    if not question:
        return jsonify({"error": "No query provided"}), 400
    
    if not namespaces or not isinstance(namespaces, list):
        return jsonify({"error": "No namespaces list provided"}), 400
    
    if len(namespaces) > 10:
        return jsonify({"error": "Máximo 10 archivos por consulta"}), 400
    
    print(f"📊 Consulta múltiple: {len(namespaces)} archivos")
    
    try:
        result = enhanced_query_service.query_multiple_files(
            query=question,
            namespaces=namespaces,
            top_k=top_k
        )
        
        if result["success"]:
            return jsonify({
                "type": "multiple_files_query",
                "query": question,
                "answer": result["consolidated_response"],
                "files_searched": result["files_searched"],
                "files_with_results": result["files_with_results"],
                "individual_responses": result["individual_responses"],
                "metadata": {
                    "analysis_type": "multi_file_enhanced"
                }
            }), 200
        else:
            return jsonify({
                "type": "multiple_files_query",
                "error": result.get("message", "Error en consulta múltiple"),
                "namespaces": namespaces,
                "query": question
            }), 404
    
    except Exception as e:
        print(f"❌ Error en consulta múltiple: {e}")
        return jsonify({
            "error": "Error procesando consulta múltiple",
            "details": str(e),
            "query": question,
            "namespaces": namespaces
        }), 500

@bp.route('/files/list', methods=['GET'])
def list_available_files():
    """Lista archivos disponibles con sus metadatos (placeholder)"""
    # TODO: Implementar un sistema de índice de archivos
    return jsonify({
        "message": "Funcionalidad de listado de archivos en desarrollo",
        "suggestion": "Guarda los namespaces de los archivos que cargas para futuras consultas"
    }), 200

@bp.route('/debug/namespaces', methods=['GET'])
def debug_namespaces():
    """Endpoint de debug para verificar namespaces disponibles"""
    try:
        from app.services.vector_database import VectorDatabaseService
        vector_db = VectorDatabaseService()
        
        # Intentar obtener información de namespaces
        index_stats = vector_db.index.describe_index_stats()
        
        namespaces_info = {}
        if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
            for namespace, stats in index_stats.namespaces.items():
                namespaces_info[namespace] = {
                    "vector_count": stats.vector_count if hasattr(stats, 'vector_count') else 0
                }
        
        return jsonify({
            "success": True,
            "total_namespaces": len(namespaces_info),
            "namespaces": namespaces_info,
            "total_vectors": index_stats.total_vector_count if hasattr(index_stats, 'total_vector_count') else 0
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error obteniendo información de namespaces"
        }), 500

@bp.route('/debug/test-query', methods=['POST'])
def debug_test_query():
    """Endpoint de debug para probar consultas sin validaciones"""
    data = request.get_json()
    question = data.get('query', 'test query')
    namespace = data.get('namespace')
    
    if not namespace:
        return jsonify({
            "error": "Se requiere namespace para debug",
            "suggestion": "Usa /api/debug/namespaces para ver namespaces disponibles"
        }), 400
    
    try:
        result = enhanced_query_service.query_specific_file(
            query=question,
            namespace=namespace,
            top_k=3
        )
        
        return jsonify({
            "debug_mode": True,
            "query": question,
            "namespace": namespace,
            "result": result
        }), 200
        
    except Exception as e:
        return jsonify({
            "debug_mode": True,
            "error": str(e),
            "query": question,
            "namespace": namespace
        }), 500
