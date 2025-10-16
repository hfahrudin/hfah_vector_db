import numpy as np
import json
import os

class VectorDB:
    def __init__(self, folder_path: str = "data", vector_file: str = "vectors.npy", meta_file: str = "metadata.json", vector_dim: int = 512):
        self.folder_path = folder_path
        self.vector_dim = vector_dim

        # Ensure folder exists
        os.makedirs(self.folder_path, exist_ok=True)

        # Full paths
        self.vector_path = os.path.join(self.folder_path, vector_file)
        self.meta_path = os.path.join(self.folder_path, meta_file)

        # Load or create files
        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()


    def _load_vectors(self) -> np.ndarray:
        """Load the numpy array from disk, or create empty if missing."""
        if os.path.exists(self.vector_path):
            return np.load(self.vector_path)
        else:
            # Create empty array with shape (0, vector_dim)
            empty_vectors = np.zeros((0, self.vector_dim))
            np.save(self.vector_path, empty_vectors)
            return empty_vectors

    def _load_metadata(self) -> dict:
        """Load the metadata dictionary from disk, or create empty if missing."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            empty_metadata = {}
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(empty_metadata, f, indent=4)
            return empty_metadata