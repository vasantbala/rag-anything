from abc import ABC, abstractmethod

class FileLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> list[tuple[str, dict]]:
        """
        Returns a list of (text, metadata) tuples.
        Each tuple represents one logical unit (page, paragraph, section).
        """
        pass