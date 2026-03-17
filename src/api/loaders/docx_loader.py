from docx import Document
from .file_loader import FileLoader

PARAGRAPHS_PER_BLOCK = 5

class DocxLoader(FileLoader):

    def load(self, file_path: str) -> list[tuple[str, dict]]:
        doc = Document(file_path)

        # Filter out empty paragraphs up front
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        results = []
        for block_start in range(0, len(paragraphs), PARAGRAPHS_PER_BLOCK):
            block = paragraphs[block_start : block_start + PARAGRAPHS_PER_BLOCK]
            text = "\n".join(block)
            results.append((text, {"paragraph_index": block_start, "source_type": "docx" }))
        
        return results