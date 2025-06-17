import numpy as np
from sentence_transformers import SentenceTransformer, util

class NLPService:
    def __init__(self):
        self.documents = []  
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, text, embedding):
        self.documents.append({"text": text, "embedding": np.array(embedding)})

    def embed_text(self, text):
        return self.model.encode(text)

    def search_similar_documents(self, query, top_k=3):
        if not self.documents:
            return []

        query_embedding = self.embed_text(query)

        similarities = []
        for doc in self.documents:
            score = util.cos_sim(query_embedding, doc["embedding"])[0][0].item()
            similarities.append((score, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_matches = [doc for _, doc in similarities[:top_k]]
        return top_matches
