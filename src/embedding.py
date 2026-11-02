"""
Tahap 2 - Penyematan Vektor (Embedding).
Mengubah teks Bahasa Ocu menjadi vektor ter-normalisasi
untuk pencarian kemiripan semantik.
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingModel:
    """Pembungkus model sentence-transformers untuk penyematan vektor."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Menginisialisasi model penyematan vektor (embedding)."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        print(f"Model loaded. Embedding dimension: {self.dimension}")
        
    def encode(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Mengubah teks menjadi penyematan vektor.
        
        Parameter:
            texts: Satu teks atau daftar teks yang akan diubah.
            show_progress: Menampilkan bilah kemajuan untuk kumpulan data besar.
            
        Mengembalikan:
            Array NumPy berisi penyematan vektor.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress and len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalisasi untuk menghitung kemiripan kosinus.
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Mengubah satu teks kueri menjadi penyematan vektor."""
        return self.encode(query, show_progress=False)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Mengubah beberapa dokumen menjadi penyematan vektor."""
        return self.encode(documents, show_progress=True)


if __name__ == "__main__":
    # Uji penyematan vektor.
    model = EmbeddingModel()
    
    test_texts = [
        "deyen poi ka kampuong",
        "saya pergi ke kampung",
        "bahasa ocu adalah bahasa daerah"
    ]
    
    embeddings = model.encode(test_texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 values): {embeddings[0][:10]}")
