import os
import uuid
import pdfplumber
import pytesseract
from PIL import Image as PILImage
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from app.utils.text_processing import chunk_text
from app.services.image_analysis_service import ImageAnalysisService
from app.services.advanced_image_analysis import AdvancedImageAnalysisService
from app.services.universal_technical_analyzer import UniversalTechnicalAnalyzer
import cv2
import numpy as np

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.image_analyzer = ImageAnalysisService()
        self.advanced_analyzer = AdvancedImageAnalysisService()  # Analizador avanzado para eléctricos
        self.universal_analyzer = UniversalTechnicalAnalyzer()   # Analizador universal

    def extract_text_from_pdf(self, file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()

    def extract_text_from_image(self, file_path):
        image = PILImage.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()

    def analyze_image_content(self, file_path):
        """Análisis básico de contenido de imagen"""
        image_cv = cv2.imread(file_path)
        return {
            "symbols": self.image_analyzer.detect_symbols(image_cv),
            "routes": self.image_analyzer.detect_routes(image_cv),
            "texts": self.image_analyzer.detect_texts(image_cv),
            "colors": self.image_analyzer.detect_dominant_colors(image_cv),
            "shapes": self.image_analyzer.detect_shapes(image_cv)
        }
    
    def analyze_electrical_plan_advanced(self, file_path, file_id):
        """Análisis avanzado específico para planos eléctricos"""
        print(f"🔬 Iniciando análisis avanzado de plano eléctrico: {file_path}")
        
        try:
            # Realizar análisis completo con el servicio avanzado
            advanced_results = self.advanced_analyzer.analyze_electrical_plan(file_path)
            
            if "error" in advanced_results:
                print(f"❌ Error en análisis avanzado: {advanced_results['error']}")
                return None
            
            # Procesar y estructurar resultados para vinculación
            processed_analysis = self._process_advanced_results(advanced_results, file_id)
            
            print(f"✅ Análisis avanzado completado para {file_path}")
            return processed_analysis
            
        except Exception as e:
            print(f"❌ Error en análisis avanzado: {e}")
            return None
    
    def analyze_technical_document_universal(self, file_path, file_id):
        """Análisis técnico universal para múltiples dominios de ingeniería"""
        print(f"🔧 Iniciando análisis técnico universal: {file_path}")
        
        try:
            # Realizar análisis completo con el analizador universal
            universal_results = self.universal_analyzer.analyze_technical_document(file_path)
            
            if "error" in universal_results:
                print(f"❌ Error en análisis universal: {universal_results['error']}")
                return None
            
            # Procesar y estructurar resultados del análisis universal
            processed_analysis = self._process_universal_results(universal_results, file_id)
            
            print(f"✅ Análisis técnico universal completado para {file_path}")
            return processed_analysis
            
        except Exception as e:
            print(f"❌ Error en análisis universal: {e}")
            return None
    
    def _process_universal_results(self, results, file_id):
        """Procesa los resultados del análisis técnico universal"""
        processed = {
            "file_id": file_id,
            "timestamp": str(uuid.uuid4()),
            "analysis_type": "universal_technical_analysis",
            "document_type": results.get("document_type", "unknown"),
            "summary": results.get("technical_assessment", {}),
            "detailed_analysis": {
                "symbols": results.get("symbols", []),
                "routes": results.get("routes", {}),
                "text_analysis": results.get("text_analysis", {}),
                "visual_analysis": results.get("visual_analysis", {})
            },
            "descriptions": {
                "comprehensive": results.get("technical_description", ""),
                "executive_summary": self._generate_universal_summary(results)
            },
            "components_detected": self._extract_universal_components(results),
            "technical_specifications": self._extract_universal_specs(results)
        }
        
        return processed
    
    def _generate_universal_summary(self, results):
        """Genera un resumen ejecutivo para análisis universal"""
        doc_type = results.get("document_type", "unknown")
        symbols_count = len(results.get("symbols", []))
        assessment = results.get("technical_assessment", {})
        complexity = assessment.get("complexity_level", "básica")
        
        # Nombres amigables para tipos de documentos
        type_names = {
            "electrical_plan": "plano eléctrico",
            "network_diagram": "diagrama de red",
            "installation_scheme": "esquema de instalación",
            "process_diagram": "diagrama de proceso",
            "architectural_plan": "plano arquitectónico",
            "mechanical_drawing": "dibujo mecánico",
            "technical_map": "mapa técnico",
            "telecom_diagram": "diagrama de telecomunicaciones",
            "infrastructure_plan": "plano de infraestructura",
            "unknown": "documento técnico"
        }
        
        doc_name = type_names.get(doc_type, "documento técnico")
        
        parts = [f"Documento identificado como {doc_name}"]
        
        if symbols_count > 0:
            parts.append(f"{symbols_count} elementos técnicos detectados")
        
        # Información de rutas/conexiones
        routes = results.get("routes", {})
        if isinstance(routes, dict) and routes.get("summary"):
            total_routes = routes["summary"].get("total_routes", 0)
            if total_routes > 0:
                parts.append(f"{total_routes} conexiones o rutas")
        
        # Información textual
        text_analysis = results.get("text_analysis", {})
        if text_analysis.get("summary"):
            specs_count = text_analysis["summary"].get("specs_count", 0)
            if specs_count > 0:
                parts.append(f"{specs_count} especificaciones técnicas")
        
        base_desc = f"{doc_name.capitalize()} con {', '.join(parts[1:])} y complejidad {complexity}."
        
        # Agregar recomendación de expertise si está disponible
        expertise = assessment.get("recommended_expertise", "")
        if expertise and len(expertise) < 100:
            base_desc += f" Recomendado para: {expertise}."
        
        return base_desc
    
    def _extract_universal_components(self, results):
        """Extrae resumen de componentes del análisis universal"""
        components = {}
        
        # Contar tipos de símbolos
        symbols = results.get("symbols", [])
        for symbol in symbols:
            component_type = symbol.get("type", "unknown")
            domain = symbol.get("domain", "generic")
            key = f"{domain}_{component_type}"
            components[key] = components.get(key, 0) + 1
        
        # Agregar información de rutas por dominio
        routes = results.get("routes", {})
        if isinstance(routes, dict) and routes.get("summary"):
            route_summary = routes["summary"]
            route_type = route_summary.get("route_type", "connections")
            components[f"total_{route_type}"] = route_summary.get("total_routes", 0)
            
            # Añadir detalles si están disponibles
            if route_summary.get("horizontal"):
                components["horizontal_connections"] = route_summary["horizontal"]
            if route_summary.get("vertical"):
                components["vertical_connections"] = route_summary["vertical"]
            if route_summary.get("intersections"):
                components["intersections"] = route_summary["intersections"]
        
        return components
    
    def _extract_universal_specs(self, results):
        """Extrae especificaciones técnicas del análisis universal"""
        specs = {}
        
        # Especificaciones de texto
        text_analysis = results.get("text_analysis", {})
        if text_analysis.get("technical_specifications"):
            specs["technical_specifications"] = [
                spec["text"] for spec in text_analysis["technical_specifications"][:10]
            ]
        
        if text_analysis.get("labels"):
            specs["labels"] = [label["text"] for label in text_analysis["labels"][:10]]
        
        if text_analysis.get("dimensions"):
            specs["dimensions"] = [dim["text"] for dim in text_analysis["dimensions"][:5]]
        
        # Análisis visual y de colores
        visual_analysis = results.get("visual_analysis", {})
        if visual_analysis.get("technical_interpretation"):
            specs["visual_interpretation"] = visual_analysis["technical_interpretation"]
        
        if visual_analysis.get("scale_analysis") and visual_analysis["scale_analysis"].get("scale_found"):
            specs["scale_indicators"] = visual_analysis["scale_analysis"]["scale_indicators"]
        
        # Tipo de documento y dominio
        specs["document_type"] = results.get("document_type", "unknown")
        
        # Evaluación de complejidad
        assessment = results.get("technical_assessment", {})
        if assessment:
            specs["complexity_assessment"] = {
                "level": assessment.get("complexity_level", "básica"),
                "total_elements": assessment.get("total_elements", 0),
                "description": assessment.get("description", ""),
                "recommended_expertise": assessment.get("recommended_expertise", "")
            }
        
        return specs
    
    def _process_advanced_results(self, results, file_id):
        """Procesa los resultados del análisis avanzado para mejor estructura"""
        processed = {
            "file_id": file_id,
            "timestamp": str(uuid.uuid4()),  # Usar como timestamp único
            "analysis_type": "advanced_electrical_plan",
            "summary": results.get("executive_summary", {}),
            "detailed_analysis": {
                "symbols": results.get("symbols", []),
                "routes": results.get("routes", {}),
                "text_analysis": results.get("text_analysis", {}),
                "color_analysis": results.get("color_analysis", {})
            },
            "descriptions": {
                "comprehensive": results.get("comprehensive_description", ""),
                "executive_summary": self._generate_short_summary(results)
            },
            "components_detected": self._extract_component_summary(results),
            "technical_specifications": self._extract_technical_specs(results)
        }
        
        return processed
    
    def _generate_short_summary(self, results):
        """Genera un resumen ejecutivo corto"""
        summary = results.get("executive_summary", {})
        parts = []
        
        if summary.get("total_symbols", 0) > 0:
            parts.append(f"{summary['total_symbols']} componentes eléctricos")
        
        if summary.get("total_routes", 0) > 0:
            parts.append(f"{summary['total_routes']} líneas de conexión")
        
        if summary.get("total_text_elements", 0) > 0:
            parts.append(f"{summary['total_text_elements']} elementos textuales")
        
        complexity = "complejidad básica"
        if summary.get("total_symbols", 0) > 20:
            complexity = "alta complejidad"
        elif summary.get("total_symbols", 0) > 10:
            complexity = "complejidad moderada"
        
        base_desc = f"Plano eléctrico con {', '.join(parts)} y {complexity}."
        
        if summary.get("key_findings"):
            findings = ", ".join(summary["key_findings"][:2])
            base_desc += f" Características destacadas: {findings}."
        
        return base_desc
    
    def _extract_component_summary(self, results):
        """Extrae un resumen de componentes detectados"""
        components = {}
        
        # Contar tipos de símbolos
        symbols = results.get("symbols", [])
        for symbol in symbols:
            component_type = symbol.get("type", "unknown")
            components[component_type] = components.get(component_type, 0) + 1
        
        # Agregar información de rutas
        routes = results.get("routes", {})
        if isinstance(routes, dict) and routes.get("summary"):
            route_summary = routes["summary"]
            components["horizontal_cables"] = route_summary.get("horizontal", 0)
            components["vertical_cables"] = route_summary.get("vertical", 0)
            components["intersections"] = route_summary.get("intersections", 0)
        
        return components
    
    def _extract_technical_specs(self, results):
        """Extrae especificaciones técnicas del análisis"""
        specs = {}
        
        # Especificaciones de texto
        text_analysis = results.get("text_analysis", {})
        if text_analysis.get("electrical_labels"):
            specs["electrical_labels"] = [label["text"] for label in text_analysis["electrical_labels"][:10]]
        
        if text_analysis.get("dimensions"):
            specs["dimensions"] = [dim["text"] for dim in text_analysis["dimensions"][:5]]
        
        # Análisis de colores principales
        color_analysis = results.get("color_analysis", {})
        if color_analysis.get("electrical_interpretation"):
            specs["color_coding"] = [
                {
                    "color": interp["color"],
                    "meaning": interp["meaning"],
                    "percentage": interp["percentage"]
                }
                for interp in color_analysis["electrical_interpretation"][:3]
            ]
        
        return specs

    def process_file(self, file_path, namespace=None, use_universal_analysis=True, analysis_mode="auto"):
        """Procesa archivo con análisis técnico universal mejorado
        
        Args:
            file_path: Ruta del archivo a procesar
            namespace: Namespace único para el archivo (se genera si no se proporciona)
            use_universal_analysis: Si usar análisis técnico universal
            analysis_mode: "auto", "universal", "electrical", "basic"
        """
        extension = os.path.splitext(file_path)[-1].lower()
        filename = os.path.basename(file_path)
        file_id = str(uuid.uuid4())
        
        # Generar namespace único para el archivo si no se proporciona
        if namespace is None:
            namespace = str(uuid.uuid4())
        
        analysis = None
        advanced_analysis = None
        universal_analysis = None
        
        print(f"📝 Procesando archivo: {filename} (ID: {file_id})")
        print(f"📞 Namespace asignado: {namespace}")
        print(f"🔧 Modo de análisis: {analysis_mode}")

        # Extracción de contenido base
        if extension == ".pdf":
            content = self.extract_text_from_pdf(file_path)
        elif extension in [".png", ".jpg", ".jpeg"]:
            content = self.extract_text_from_image(file_path)
            
            # Análisis básico de imagen (siempre se ejecuta)
            analysis = self.analyze_image_content(file_path)
            
            # Determinar tipo de análisis a usar
            if analysis_mode == "auto" and use_universal_analysis:
                # Modo automático: usar análisis universal como principal
                universal_analysis = self.analyze_technical_document_universal(file_path, file_id)
                
                # Si es un plano eléctrico específicamente, también usar análisis especializado
                if (universal_analysis and 
                    universal_analysis.get("document_type") == "electrical_plan"):
                    print("🔌 Detectado plano eléctrico, ejecutando análisis especializado adicional...")
                    advanced_analysis = self.analyze_electrical_plan_advanced(file_path, file_id)
            
            elif analysis_mode == "universal":
                # Solo análisis universal
                universal_analysis = self.analyze_technical_document_universal(file_path, file_id)
            
            elif analysis_mode == "electrical":
                # Solo análisis eléctrico especializado
                advanced_analysis = self.analyze_electrical_plan_advanced(file_path, file_id)
            
            # analysis_mode == "basic" usa solo el análisis básico
            
            # Construir contenido enriquecido
            content = self._build_enriched_content_universal(
                content, analysis, advanced_analysis, universal_analysis
            )
            
        else:
            raise ValueError("Unsupported file type")

        print(f"✅ Texto extraído: {len(content)} caracteres")

        # Crear embeddings con información de contexto mejorada
        vectors = self._create_enhanced_embeddings_universal(
            content, filename, file_id, namespace, analysis, advanced_analysis, universal_analysis
        )

        # Subir a Pinecone con namespace específico
        self.index.upsert(vectors=vectors, namespace=namespace)
        print(f"✅ Embeddings cargados a Pinecone: {len(vectors)} (namespace: {namespace})")

        # Preparar respuesta completa
        processing_result = {
            "file_id": file_id,
            "namespace": namespace,
            "filename": filename,
            "content_length": len(content),
            "chunks_created": len(vectors),
            "analysis_mode": analysis_mode,
            "basic_analysis": analysis,
            "advanced_analysis": advanced_analysis,
            "universal_analysis": universal_analysis,
            "processing_timestamp": str(uuid.uuid4())
        }

        return content, vectors, processing_result
    
    def _build_enriched_content(self, base_content, basic_analysis, advanced_analysis):
        """Construye contenido enriquecido con información de análisis"""
        content_parts = [base_content]
        
        # Información del análisis básico
        if basic_analysis:
            basic_info = []
            
            if basic_analysis.get("symbols"):
                symbol_names = [s["type"] for s in basic_analysis["symbols"]]
                basic_info.append("Símbolos detectados: " + ", ".join(symbol_names))

            if basic_analysis.get("texts"):
                text_words = [t["text"] for t in basic_analysis["texts"]]
                basic_info.append("Textos visibles: " + ", ".join(text_words[:10]))

            if basic_analysis.get("colors"):
                main_color = basic_analysis["colors"][0]["rgb"]
                basic_info.append(f"Color predominante: RGB {main_color}")
            
            if basic_info:
                content_parts.append("\n=== ANÁLISIS BÁSICO ===")
                content_parts.extend(basic_info)
        
        # Información del análisis avanzado
        if advanced_analysis:
            content_parts.append("\n=== ANÁLISIS AVANZADO ===")
            
            # Resumen ejecutivo
            if advanced_analysis.get("descriptions", {}).get("executive_summary"):
                content_parts.append(f"Resumen: {advanced_analysis['descriptions']['executive_summary']}")
            
            # Componentes detectados
            if advanced_analysis.get("components_detected"):
                components = advanced_analysis["components_detected"]
                component_list = [f"{k}: {v}" for k, v in components.items() if v > 0]
                if component_list:
                    content_parts.append("Componentes: " + ", ".join(component_list))
            
            # Especificaciones técnicas
            if advanced_analysis.get("technical_specifications"):
                specs = advanced_analysis["technical_specifications"]
                if specs.get("electrical_labels"):
                    content_parts.append("Etiquetas eléctricas: " + ", ".join(specs["electrical_labels"][:5]))
                
                if specs.get("color_coding"):
                    color_meanings = [cc["meaning"] for cc in specs["color_coding"][:3]]
                    content_parts.append("Codificación de colores: " + ", ".join(color_meanings))
        
        return "\n".join(content_parts)
    
    def _create_enhanced_embeddings(self, content, filename, file_id, namespace, basic_analysis, advanced_analysis):
        """Crea embeddings enriquecidos con metadatos completos"""
        chunks = chunk_text(content, max_length=400)  # Chunks un poco más grandes
        print(f"✅ Texto dividido en {len(chunks)} fragmentos")

        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            
            # Metadatos básicos
            metadata = {
                "text": chunk,
                "filename": filename,
                "file_id": file_id,
                "chunk_index": i,
                "namespace": namespace,
                "content_type": "electrical_plan" if advanced_analysis else "general_image",
                "analysis_version": "enhanced_v1.5"
            }
            
            # Añadir metadatos del análisis básico
            if basic_analysis:
                metadata["basic_symbols_count"] = len(basic_analysis.get("symbols", []))
                metadata["basic_colors_count"] = len(basic_analysis.get("colors", []))
                metadata["basic_shapes_count"] = len(basic_analysis.get("shapes", []))
            
            # Añadir metadatos del análisis avanzado
            if advanced_analysis:
                summary = advanced_analysis.get("summary", {})
                metadata["advanced_symbols_count"] = summary.get("total_symbols", 0)
                metadata["advanced_routes_count"] = summary.get("total_routes", 0)
                metadata["advanced_text_elements"] = summary.get("total_text_elements", 0)
                metadata["complexity_level"] = self._get_complexity_level(summary)
                
                # Key findings como string
                if summary.get("key_findings"):
                    metadata["key_findings"] = "; ".join(summary["key_findings"][:3])
                
                # Componentes principales
                components = advanced_analysis.get("components_detected", {})
                top_components = sorted(components.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_components:
                    metadata["main_components"] = "; ".join([f"{k}:{v}" for k, v in top_components])
            
            vectors.append({
                "id": f"{file_id}_chunk_{i}",
                "values": embedding,
                "metadata": metadata
            })

        return vectors
    
    def _get_complexity_level(self, summary):
        """Determina el nivel de complejidad basado en el resumen"""
        total_elements = summary.get("total_symbols", 0) + summary.get("total_routes", 0)
        
        if total_elements >= 50:
            return "muy_alta"
        elif total_elements >= 30:
            return "alta"
        elif total_elements >= 15:
            return "moderada"
        else:
            return "basica"
    
    def _build_enriched_content_universal(self, base_content, basic_analysis, advanced_analysis, universal_analysis):
        """Construye contenido enriquecido con información de análisis múltiple"""
        content_parts = [base_content]
        
        # Información del análisis básico
        if basic_analysis:
            basic_info = []
            
            if basic_analysis.get("symbols"):
                symbol_names = [s["type"] for s in basic_analysis["symbols"]]
                basic_info.append("Símbolos básicos: " + ", ".join(symbol_names[:5]))

            if basic_analysis.get("texts"):
                text_words = [t["text"] for t in basic_analysis["texts"]]
                basic_info.append("Textos básicos: " + ", ".join(text_words[:5]))
            
            if basic_info:
                content_parts.append("\n=== ANÁLISIS BÁSICO ===")
                content_parts.extend(basic_info)
        
        # Información del análisis universal (prioritario)
        if universal_analysis:
            content_parts.append("\n=== ANÁLISIS TÉCNICO UNIVERSAL ===")
            
            # Tipo de documento
            doc_type = universal_analysis.get("document_type", "unknown")
            if doc_type != "unknown":
                content_parts.append(f"Tipo de documento: {doc_type}")
            
            # Resumen ejecutivo
            if universal_analysis.get("descriptions", {}).get("executive_summary"):
                content_parts.append(f"Resumen: {universal_analysis['descriptions']['executive_summary']}")
            
            # Componentes detectados por dominio
            if universal_analysis.get("components_detected"):
                components = universal_analysis["components_detected"]
                component_list = [f"{k}: {v}" for k, v in components.items() if v > 0]
                if component_list:
                    content_parts.append("Componentes técnicos: " + ", ".join(component_list[:8]))
            
            # Especificaciones técnicas
            if universal_analysis.get("technical_specifications"):
                specs = universal_analysis["technical_specifications"]
                if specs.get("technical_specifications"):
                    content_parts.append("Especificaciones: " + ", ".join(specs["technical_specifications"][:5]))
                
                if specs.get("labels"):
                    content_parts.append("Etiquetas: " + ", ".join(specs["labels"][:5]))
                
                if specs.get("visual_interpretation"):
                    content_parts.append("Interpretación visual: " + ", ".join(specs["visual_interpretation"][:3]))
            
            # Información de complejidad
            if universal_analysis.get("summary", {}).get("description"):
                complexity_desc = universal_analysis["summary"]["description"]
                if len(complexity_desc) < 200:
                    content_parts.append(f"Evaluación: {complexity_desc}")
        
        # Información del análisis eléctrico especializado (si está disponible)
        if advanced_analysis:
            content_parts.append("\n=== ANÁLISIS ESPECIALIZADO ELÉCTRICO ===")
            
            # Resumen ejecutivo del análisis eléctrico
            if advanced_analysis.get("descriptions", {}).get("executive_summary"):
                content_parts.append(f"Análisis eléctrico: {advanced_analysis['descriptions']['executive_summary']}")
            
            # Especificaciones eléctricas específicas
            if advanced_analysis.get("technical_specifications"):
                specs = advanced_analysis["technical_specifications"]
                if specs.get("electrical_labels"):
                    content_parts.append("Componentes eléctricos: " + ", ".join(specs["electrical_labels"][:5]))
                
                if specs.get("color_coding"):
                    color_meanings = [cc["meaning"] for cc in specs["color_coding"][:3]]
                    content_parts.append("Códigos de color eléctricos: " + ", ".join(color_meanings))
        
        return "\n".join(content_parts)
    
    def _create_enhanced_embeddings_universal(self, content, filename, file_id, namespace, basic_analysis, advanced_analysis, universal_analysis):
        """Crea embeddings enriquecidos con metadatos de análisis múltiple"""
        chunks = chunk_text(content, max_length=450)  # Chunks más grandes para más contexto
        print(f"✅ Texto dividido en {len(chunks)} fragmentos")

        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            
            # Metadatos básicos
            metadata = {
                "text": chunk,
                "filename": filename,
                "file_id": file_id,
                "chunk_index": i,
                "namespace": namespace,
                "analysis_version": "universal_v2.0"
            }
            
            # Determinar tipo de contenido principal
            content_type = "general_image"
            if universal_analysis:
                content_type = universal_analysis.get("document_type", "unknown_technical")
            elif advanced_analysis:
                content_type = "electrical_plan"
            
            metadata["content_type"] = content_type
            
            # Añadir metadatos del análisis básico
            if basic_analysis:
                metadata["basic_symbols_count"] = len(basic_analysis.get("symbols", []))
                metadata["basic_colors_count"] = len(basic_analysis.get("colors", []))
                metadata["basic_shapes_count"] = len(basic_analysis.get("shapes", []))
            
            # Añadir metadatos del análisis universal (prioritario)
            if universal_analysis:
                summary = universal_analysis.get("summary", {})
                metadata["universal_symbols_count"] = summary.get("symbols_count", 0)
                metadata["universal_routes_count"] = summary.get("routes_count", 0)
                metadata["universal_text_elements"] = summary.get("text_elements", 0)
                metadata["complexity_level"] = summary.get("complexity_level", "basica")
                metadata["technical_domain"] = summary.get("technical_domain", "unknown")
                
                # Expertise recomendado
                if summary.get("recommended_expertise"):
                    expertise = summary["recommended_expertise"]
                    if len(expertise) < 100:
                        metadata["recommended_expertise"] = expertise
                
                # Componentes principales
                components = universal_analysis.get("components_detected", {})
                if components:
                    top_components = sorted(components.items(), key=lambda x: x[1], reverse=True)[:5]
                    if top_components:
                        metadata["main_technical_components"] = "; ".join([f"{k}:{v}" for k, v in top_components])
                
                # Especificaciones técnicas clave
                specs = universal_analysis.get("technical_specifications", {})
                if specs.get("document_type"):
                    metadata["document_type_detected"] = specs["document_type"]
                
                if specs.get("technical_specifications"):
                    tech_specs = specs["technical_specifications"][:3]
                    if tech_specs:
                        metadata["key_technical_specs"] = "; ".join(tech_specs)
            
            # Añadir metadatos del análisis eléctrico especializado (complementario)
            if advanced_analysis:
                elec_summary = advanced_analysis.get("summary", {})
                metadata["electrical_symbols_count"] = elec_summary.get("total_symbols", 0)
                metadata["electrical_routes_count"] = elec_summary.get("total_routes", 0)
                
                # Características eléctricas específicas
                if advanced_analysis.get("technical_specifications", {}).get("electrical_labels"):
                    elec_labels = advanced_analysis["technical_specifications"]["electrical_labels"][:3]
                    if elec_labels:
                        metadata["electrical_components"] = "; ".join(elec_labels)
            
            vectors.append({
                "id": f"{file_id}_chunk_{i}",
                "values": embedding,
                "metadata": metadata
            })

        return vectors
