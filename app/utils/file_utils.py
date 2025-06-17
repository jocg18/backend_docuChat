import pdfplumber

def extract_text_from_pdf(file_path: str) -> str:
    """Extrae texto de un archivo PDF usando pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Error al leer el PDF: {e}")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Lee el contenido de un archivo de texto plano."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo TXT: {e}")

