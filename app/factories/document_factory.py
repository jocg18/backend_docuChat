from app.services.document_processor import PdfProcessor, ImageProcessor

class DocumentFactory:
    @staticmethod
    def get_processor(filetype):
        if filetype == 'pdf':
            return PdfProcessor()
        elif filetype in ['png', 'jpg', 'jpeg']:
            return ImageProcessor()
        else:
            raise ValueError("Unsupported file type")
