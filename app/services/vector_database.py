# backend/app/services/vector_database.py
import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger(__name__)

class VectorDatabaseService:
    def __init__(self):
        api_key     = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name  = os.getenv("PINECONE_INDEX_NAME")

        if not api_key:
            logger.error("PINECONE_API_KEY no configurada")
            raise ValueError("PINECONE_API_KEY no configurada")

        self.pc = Pinecone(api_key=api_key)

        existing = self.pc.list_indexes().names()
        if index_name not in existing:
            logger.info(f"Creando índice '{index_name}' en Pinecone...")
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment)
            )

        self.index = self.pc.Index(index_name)
        logger.info(f"Conectado al índice Pinecone: {index_name}")

    def upsert_document(self, doc_id: str, embeddings: list, metadata: dict, namespace: str):
        """
        Inserta vectores en Pinecone en un namespace específico.
        """
        self.index.upsert(vectors=[{
            "id": doc_id,
            "values": embeddings,
            "metadata": metadata
        }], namespace=namespace)

    def buscar_similitud(self, pregunta: str, embedding_model, top_k: int = 5, namespace: str = None):
        """
        Genera embedding de la pregunta y consulta Pinecone en el namespace dado (si se especifica).
        """
        query_embedding = embedding_model.encode(pregunta).tolist()
        logger.info(f"Consultando Pinecone con embedding generado...")

        query_kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }

        if namespace:
            query_kwargs["namespace"] = namespace  # solo se agrega si no es None

        result = self.index.query(**query_kwargs)

        matches = getattr(result, "matches", result.get("matches", []))
        logger.info(f"{len(matches)} resultados obtenidos de Pinecone")

        return [
            {
                "text": m.metadata.get("text", ""),
                "score": getattr(m, "score", m.get("score", 0))
            }
            for m in matches
        ]

