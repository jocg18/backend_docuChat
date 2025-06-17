class Embedding:
    def __init__(self, vector_id: str, values: list, metadata: dict = None):
        self.vector_id = vector_id
        self.values = values  
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "id": self.vector_id,
            "values": self.values,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            vector_id=data.get("id"),
            values=data.get("values"),
            metadata=data.get("metadata")
        )
