import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from schemas import *

class VectorDB:
    def __init__(self, folder_path: str = "data", vector_file: str = "vectors.npy", meta_file: str = "metadata.json"):
        self.folder_path = folder_path

        # Ensure folder exists
        os.makedirs(self.folder_path, exist_ok=True)

        # Full paths
        self.vector_path = os.path.join(self.folder_path, vector_file)
        self.meta_path = os.path.join(self.folder_path, meta_file)

        print("Loading embedding model...")
        # Load MiniLM embedding model
        self.vector_dim = 384 # maximum for minilm
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load or create files
        print("Loading vector database...")
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
        
    def add_data(self, request: AddRequest) -> AddResponse:
        """Add a new text entry with metadata to the vector database."""
        content = request.text
        label = request.label
        source = request.source if request.source else ""

        vector = self.model.encode([content], convert_to_numpy=True)[0].reshape(1, -1)  # 2D shape (1, dim)

        self.vectors = np.vstack([self.vectors, vector])
        np.save(self.vector_path, self.vectors)

        new_id = str(len(self.metadata))  # incremental ID
        meta = {
            "text": content,
            "label": label,
            "source": source
        }
        
        self.metadata[new_id] = meta
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

        return AddResponse(status="success", vector_index=int(new_id))
    
    def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Search the vector database for the most similar entries to the query."""
        query = request.query
        top_k = request.top_k

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        scores = self._cosine_sim(query_embedding)

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "metadata": self.metadata.get(str(idx), {})
            })

        return InvokeResponse(results=results)
    

    def _cosine_sim(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query embedding and all stored vectors.
        
        Returns:
            scores: np.ndarray of shape (num_vectors,), higher means more similar.
        """
        if self.vectors.shape[0] == 0:
            return np.array([])

        # Normalize stored vectors and query
        vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Compute cosine similarity
        scores = vectors_norm @ query_norm  # Dot product gives cosine similarity
        return scores
    
    def _cosine_sim_scratch(self, query_embedding: np.ndarray) -> np.ndarray:
        pass