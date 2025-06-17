from datetime import datetime

class Document:
    def __init__(self, doc_id: str, filename: str, upload_date: datetime = None, metadata: dict = None):
        self.doc_id = doc_id
        self.filename = filename
        self.upload_date = upload_date or datetime.utcnow()
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "upload_date": self.upload_date.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict):
        upload_date = datetime.fromisoformat(data["upload_date"]) if "upload_date" in data else None
        return cls(
            doc_id=data.get("doc_id"),
            filename=data.get("filename"),
            upload_date=upload_date,
            metadata=data.get("metadata")
        )
