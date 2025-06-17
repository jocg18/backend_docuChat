"""Servicio de consultas mejorado con vinculación específica por archivo

Este servicio proporciona:
- Búsquedas vinculadas a archivos específicos mediante namespaces
- Respuestas contextualizadas con análisis avanzado
- Filtrado inteligente por tipo de contenido
- Generación de respuestas enriquecidas
"""

import os
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class EnhancedQueryService:
    def __init__(self):
        """Inicializa el servicio de consultas mejorado"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    def query_specific_file(self, query: str, namespace: str, top_k: int = 5) -> Dict:
        """Realiza consulta específica a un archivo usando su namespace"""
        try:
            print(f"🔍 Consultando en namespace específico: {namespace}")
            
            # Generar embedding de la consulta
            query_embedding = self.model.encode(query).tolist()
            
            # Buscar en el namespace específico
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            print(f"📊 Resultados obtenidos: {len(results.matches) if results.matches else 0} matches")
            
            if not results.matches:
                # Verificar si el namespace existe
                try:
                    index_stats = self.index.describe_index_stats()
                    namespace_exists = False
                    if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
                        namespace_exists = namespace in index_stats.namespaces
                    
                    if not namespace_exists:
                        return {
                            "success": False,
                            "message": f"El archivo con namespace '{namespace}' no existe en la base de datos",
                            "namespace": namespace,
                            "query": query,
                            "error_type": "namespace_not_found"
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"No se encontraron resultados relevantes para la consulta en este archivo",
                            "namespace": namespace,
                            "query": query,
                            "error_type": "no_relevant_matches",
                            "suggestion": "Intenta reformular la pregunta o usar términos más generales"
                        }
                except Exception as stats_error:
                    print(f"⚠️ Error verificando namespace: {stats_error}")
                    return {
                        "success": False,
                        "message": "No se encontraron resultados en el archivo especificado",
                        "namespace": namespace,
                        "query": query,
                        "error_type": "no_matches"
                    }
            
            # Procesar resultados con contexto del archivo
            processed_results = self._process_file_specific_results(results, query, namespace)
            
            return {
                "success": True,
                "namespace": namespace,
                "query": query,
                "results_count": len(results.matches),
                "response": processed_results["response"],
                "confidence": processed_results["confidence"],
                "file_context": processed_results["file_context"],
                "relevant_chunks": processed_results["chunks"][:3]  # Top 3 chunks
            }
            
        except Exception as e:
            print(f"❌ Error en consulta específica: {e}")
            return {
                "success": False,
                "error": str(e),
                "namespace": namespace,
                "query": query
            }
    
    def query_multiple_files(self, query: str, namespaces: List[str], top_k: int = 3) -> Dict:
        """Realiza consulta en múltiples archivos específicos"""
        try:
            print(f"🔍 Consultando en {len(namespaces)} archivos específicos")
            
            all_results = []
            file_responses = {}
            
            # Consultar cada archivo
            for namespace in namespaces:
                file_result = self.query_specific_file(query, namespace, top_k)
                if file_result["success"]:
                    all_results.extend(file_result["relevant_chunks"])
                    file_responses[namespace] = {
                        "filename": file_result.get("file_context", {}).get("filename", "Desconocido"),
                        "response": file_result["response"],
                        "confidence": file_result["confidence"]
                    }
            
            if not all_results:
                return {
                    "success": False,
                    "message": "No se encontraron resultados en ningún archivo especificado",
                    "namespaces": namespaces,
                    "query": query
                }
            
            # Generar respuesta consolidada
            consolidated_response = self._consolidate_multi_file_response(query, file_responses)
            
            return {
                "success": True,
                "query": query,
                "files_searched": len(namespaces),
                "files_with_results": len(file_responses),
                "consolidated_response": consolidated_response,
                "individual_responses": file_responses
            }
            
        except Exception as e:
            print(f"❌ Error en consulta múltiple: {e}")
            return {
                "success": False,
                "error": str(e),
                "namespaces": namespaces,
                "query": query
            }
    
    def intelligent_search(self, query: str, content_filter: Optional[str] = None, 
                          complexity_filter: Optional[str] = None, top_k: int = 10) -> Dict:
        """Búsqueda inteligente con filtros avanzados"""
        try:
            print(f"📚 Realizando búsqueda inteligente con filtros")
            
            # Generar embedding de la consulta
            query_embedding = self.model.encode(query).tolist()
            
            # Construir filtros
            filter_dict = {}
            if content_filter:
                filter_dict["content_type"] = content_filter
            if complexity_filter:
                filter_dict["complexity_level"] = complexity_filter
            
            # Realizar búsqueda
            search_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter_dict:
                search_params["filter"] = filter_dict
            
            results = self.index.query(**search_params)
            
            if not results.matches:
                return {
                    "success": False,
                    "message": "No se encontraron resultados con los filtros especificados",
                    "query": query,
                    "filters_applied": filter_dict
                }
            
            # Agrupar resultados por archivo
            grouped_results = self._group_results_by_file(results)
            
            # Generar respuesta inteligente
            intelligent_response = self._generate_intelligent_response(query, grouped_results)
            
            return {
                "success": True,
                "query": query,
                "filters_applied": filter_dict,
                "files_found": len(grouped_results),
                "total_matches": len(results.matches),
                "intelligent_response": intelligent_response,
                "file_summaries": self._create_file_summaries(grouped_results)
            }
            
        except Exception as e:
            print(f"❌ Error en búsqueda inteligente: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _process_file_specific_results(self, results, query: str, namespace: str) -> Dict:
        """Procesa resultados específicos de un archivo"""
        matches = results.matches
        if not matches:
            return {"response": "No se encontró información relevante", "confidence": 0.0}
        
        # Extraer información del archivo
        first_match = matches[0]
        metadata = first_match.metadata
        
        filename = metadata.get("filename", "Archivo desconocido")
        file_id = metadata.get("file_id", "ID desconocido")
        content_type = metadata.get("content_type", "general")
        analysis_version = metadata.get("analysis_version", "básica")
        
        # Construir contexto del archivo
        file_context = {
            "filename": filename,
            "file_id": file_id,
            "content_type": content_type,
            "analysis_version": analysis_version,
            "namespace": namespace
        }
        
        # Añadir información avanzada si está disponible
        if content_type == "electrical_plan":
            file_context.update({
                "symbols_count": metadata.get("advanced_symbols_count", 0),
                "routes_count": metadata.get("advanced_routes_count", 0),
                "complexity": metadata.get("complexity_level", "desconocida"),
                "key_findings": metadata.get("key_findings", ""),
                "main_components": metadata.get("main_components", "")
            })
        
        # Combinar contenido relevante
        relevant_texts = [match.metadata.get("text", "") for match in matches[:3]]
        combined_content = " ".join(relevant_texts)
        
        # Calcular confianza promedio
        avg_confidence = sum(match.score for match in matches) / len(matches)
        
        # Generar respuesta contextualizada
        response = self._generate_contextualized_response(
            query, combined_content, file_context, avg_confidence
        )
        
        return {
            "response": response,
            "confidence": avg_confidence,
            "file_context": file_context,
            "chunks": [
                {
                    "text": match.metadata.get("text", "")[:500] + "..." if len(match.metadata.get("text", "")) > 500 else match.metadata.get("text", ""),
                    "score": match.score,
                    "chunk_index": match.metadata.get("chunk_index", 0)
                }
                for match in matches
            ]
        }
    
    def _generate_contextualized_response(self, query: str, content: str, 
                                        file_context: Dict, confidence: float) -> str:
        """Genera respuesta contextualizada según el tipo de archivo"""
        filename = file_context.get("filename", "el archivo")
        content_type = file_context.get("content_type", "general")
        
        # Respuesta base
        response_parts = []
        
        # Encabezado contextual
        if content_type == "electrical_plan":
            complexity = file_context.get("complexity", "desconocida")
            symbols_count = file_context.get("symbols_count", 0)
            
            response_parts.append(
                f"Según el análisis del plano eléctrico '{filename}' "
                f"(complejidad {complexity}, {symbols_count} componentes detectados):"
            )
            
            # Añadir información de componentes si está disponible
            if file_context.get("main_components"):
                components = file_context["main_components"].replace(";", ", ")
                response_parts.append(f"Componentes principales: {components}.")
            
        else:
            response_parts.append(f"Según el contenido de '{filename}':")
        
        # Contenido principal (primeras oraciones relevantes)
        sentences = content.split(".")[:3]
        main_content = ". ".join(sentences)
        if main_content:
            response_parts.append(main_content)
        
        # Indicador de confianza
        if confidence > 0.8:
            confidence_text = "Información muy relevante encontrada."
        elif confidence > 0.6:
            confidence_text = "Información relevante encontrada."
        else:
            confidence_text = "Información parcialmente relevante."
        
        response_parts.append(confidence_text)
        
        # Nota sobre contexto específico del archivo
        if file_context.get("key_findings"):
            findings = file_context["key_findings"].replace(";", ", ")
            response_parts.append(f"Características destacadas del archivo: {findings}.")
        
        return " ".join(response_parts)
    
    def _consolidate_multi_file_response(self, query: str, file_responses: Dict) -> str:
        """Consolida respuestas de múltiples archivos"""
        if not file_responses:
            return "No se encontró información relevante en los archivos especificados."
        
        response_parts = []
        response_parts.append(f"Información encontrada en {len(file_responses)} archivo(s):")
        
        for namespace, file_data in file_responses.items():
            filename = file_data.get("filename", "Archivo desconocido")
            response = file_data.get("response", "")
            confidence = file_data.get("confidence", 0.0)
            
            # Resumir respuesta si es muy larga
            if len(response) > 200:
                response = response[:200] + "..."
            
            confidence_indicator = "🔴" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟢"
            
            response_parts.append(
                f"\n{confidence_indicator} **{filename}**: {response}"
            )
        
        response_parts.append(
            "\n¿Necesitas información más detallada de alguno de estos archivos?"
        )
        
        return "\n".join(response_parts)
    
    def _group_results_by_file(self, results) -> Dict:
        """Agrupa resultados por archivo"""
        grouped = {}
        
        for match in results.matches:
            namespace = match.metadata.get("namespace", "unknown")
            filename = match.metadata.get("filename", "Archivo desconocido")
            
            if namespace not in grouped:
                grouped[namespace] = {
                    "filename": filename,
                    "matches": [],
                    "metadata_summary": self._extract_file_metadata_summary(match.metadata)
                }
            
            grouped[namespace]["matches"].append(match)
        
        return grouped
    
    def _extract_file_metadata_summary(self, metadata: Dict) -> Dict:
        """Extrae resumen de metadatos del archivo"""
        return {
            "content_type": metadata.get("content_type", "general"),
            "complexity": metadata.get("complexity_level", "desconocida"),
            "symbols_count": metadata.get("advanced_symbols_count", 0),
            "routes_count": metadata.get("advanced_routes_count", 0),
            "analysis_version": metadata.get("analysis_version", "básica")
        }
    
    def _generate_intelligent_response(self, query: str, grouped_results: Dict) -> str:
        """Genera respuesta inteligente basada en resultados agrupados"""
        if not grouped_results:
            return "No se encontraron resultados relevantes."
        
        response_parts = []
        response_parts.append(f"Encontré información relevante en {len(grouped_results)} archivo(s):")
        
        # Analizar tipos de archivos encontrados
        electrical_plans = 0
        pdf_files = 0
        general_images = 0
        
        for namespace, file_data in grouped_results.items():
            filename = file_data["filename"].lower()
            if file_data["metadata_summary"]["content_type"] == "electrical_plan":
                electrical_plans += 1
            elif filename.endswith('.pdf'):
                pdf_files += 1
            else:
                general_images += 1
        
        # Descripción de tipos de contenido
        if electrical_plans > 0:
            response_parts.append(
                f"\n📊 {electrical_plans} plano(s) eléctrico(s) con análisis avanzado"
            )
        
        if pdf_files > 0:
            response_parts.append(
                f"\n📄 {pdf_files} PDF(s) con análisis básico"
            )
        
        if general_images > 0:
            response_parts.append(
                f"\n🖼️ {general_images} imagen(es) con análisis básico"
            )
        
        # Resumen de cada archivo con mejor coincidencia
        response_parts.append("\n\n**Resumen por archivo:**")
        
        for namespace, file_data in list(grouped_results.items())[:3]:  # Top 3 archivos
            filename = file_data["filename"]
            best_match = file_data["matches"][0]  # Mejor coincidencia
            score = best_match.score
            text_snippet = best_match.metadata.get("text", "")[:800] + "..." if len(best_match.metadata.get("text", "")) > 800 else best_match.metadata.get("text", "")
            
            confidence_emoji = "🔴" if score > 0.8 else "🟡" if score > 0.6 else "🟢"
            
            response_parts.append(
                f"\n{confidence_emoji} **{filename}** (relevancia: {score:.2f})\n{text_snippet}"
            )
        
        return "\n".join(response_parts)
    
    def _create_file_summaries(self, grouped_results: Dict) -> List[Dict]:
        """Crea resúmenes de archivos encontrados"""
        summaries = []
        
        for namespace, file_data in grouped_results.items():
            summary = {
                "namespace": namespace,
                "filename": file_data["filename"],
                "matches_count": len(file_data["matches"]),
                "best_score": file_data["matches"][0].score if file_data["matches"] else 0.0,
                "metadata_summary": file_data["metadata_summary"]
            }
            
            # Añadir snippet del mejor resultado
            if file_data["matches"]:
                best_text = file_data["matches"][0].metadata.get("text", "")
                summary["preview"] = best_text[:600] + "..." if len(best_text) > 600 else best_text
            
            summaries.append(summary)
        
        # Ordenar por relevancia
        summaries.sort(key=lambda x: x["best_score"], reverse=True)
        
        return summaries

