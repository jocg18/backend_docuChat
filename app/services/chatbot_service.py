# backend/app/services/chatbot_service.py
import logging
import re
from sentence_transformers import SentenceTransformer
from app.services.vector_database import VectorDatabaseService

logger = logging.getLogger(__name__)


class ChatbotService:
    def __init__(self):
        self.vector_db = VectorDatabaseService()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    
    #  MÉTODO CORRECTAMENTE INDENTADO
    
    def answer_query(self, question: str, file_id: str = None) -> str:
        logger.info(f"Realizando búsqueda semántica con la pregunta: {question}")

        try:
            matches = self.vector_db.buscar_similitud(
                pregunta=question,
                embedding_model=self.embedding_model,
                top_k=5,
                namespace=file_id,
            )
        except Exception as e:
            logger.error(f"Error al consultar Pinecone: {e}")
            return "Error al procesar la consulta con la base de datos vectorial."

        if not matches:
            return "No se encontró información relevante en los documentos cargados."

        # Limpiar y normalizar fragmentos
        vistos = set()
        fragmentos_limpios = []
        for m in matches:
            texto = m.get("text", "").strip()
            if not texto:
                continue
            texto = re.sub(r"^\d+\.\s*", "", texto)
            texto = re.sub(r"\s+", " ", texto).strip()

            if texto.lower() not in vistos:
                fragmentos_limpios.append(texto)
                vistos.add(texto.lower())

        if not fragmentos_limpios:
            return "Los fragmentos encontrados no contienen información útil."

        # Construir respuesta natural
        conectores = [
            "Además", "También", "Por otro lado",
            "Cabe destacar", "En este sentido"
        ]
        respuesta = "Según el contenido del documento, se puede concluir lo siguiente:\n\n"
        for i, frag in enumerate(fragmentos_limpios):
            if i == 0:
                respuesta += frag
            else:
                con = conectores[i % len(conectores)]
                respuesta += f"\n{con}, {frag[0].lower() + frag[1:]}"

        return respuesta
