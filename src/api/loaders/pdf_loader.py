import pdfplumber
from src.api.loaders.file_loader import FileLoader

class PdfLoader(FileLoader):
    def load(self, file_path: str) -> list[tuple[str, dict]]:
        results = []
        
        #pdfplumber.open() returns a context manager
        #.pages is a lit of Page objects
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # .extract_text() returns a str or None
                # returns None if the page has no extractable text (e.g. scanned image)
                text = page.extract_text()
                
                #skip pages with no text
                if not text or not text.strip():
                    continue
                
                #page.page_number is 1-based
                metadata = {
                    "page_number": page.page_number,
                    "source_type": "pdf",
                }
                results.append((text.strip(), metadata))
                
        return results