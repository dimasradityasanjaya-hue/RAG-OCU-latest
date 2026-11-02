"""
Tahap 3 - Pembangunan Basis Data Vektor dan Indeks FAISS.
Membangun indeks vektor, menyimpan metadata, dan memuat artefak
indeks agar siap dipakai pada runtime penerjemahan.
"""
import faiss
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence
from pathlib import Path

from src.config import (
    FAISS_INDEX_FILE, 
    METADATA_FILE, 
)
from src.embedding import EmbeddingModel
from src.data_loader import DataLoader, TranslationEntry


class VectorStore:
    """Basis data vektor FAISS untuk pencarian semantik."""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """Menginisialisasi basis data vektor."""
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []
        self.entries: List[TranslationEntry] = []
        self.load_error: Optional[Exception] = None
        

    def build_index(self, entries: Sequence[TranslationEntry]) -> None:
        """
        Membangun indeks FAISS dari entri terjemahan.
        
        Parameter:
            entries: Daftar objek TranslationEntry.
        """
        self.entries = list(entries)
        self.metadata = [entry.to_dict() for entry in self.entries]

        if not self.entries:
            raise ValueError("Cannot build a vector store from empty entries.")
        
        # Buat dokumen untuk penyematan vektor menggunakan teks Bahasa Ocu
        # agar kueri Bahasa Ocu dapat menemukan kecocokan dengan baik.
        texts = [entry.bahasa_ocu for entry in self.entries]
        
        print(f"Creating embeddings for {len(texts)} entries...")
        embeddings = np.asarray(
            self.embedding_model.encode_documents(texts),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.entries):
            raise ValueError(
                "Embedding model must return one 2D vector for each entry."
            )
        if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must be finite and have a positive dimension.")
        
        # Buat indeks FAISS (Inner Product pada vektor ternormalisasi = kemiripan kosinus).
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Tambahkan vektor ke indeks FAISS. 115
        self.index.add(embeddings.astype(np.float32))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
        
    def save(self, index_path: Path = FAISS_INDEX_FILE, metadata_path: Path = METADATA_FILE) -> None:
        """Menyimpan indeks dan metadata ke disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
            
        # Indeks dan metadata dapat disimpan di direktori yang berbeda.
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan indeks FAISS
        faiss.write_index(self.index, str(index_path))
        print(f"Index saved to {index_path}")
        
        # Simpan metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'entries': [e.to_dict() for e in self.entries]
            }, f)
        print(f"Metadata saved to {metadata_path}")
        
    def load(self, index_path: Path = FAISS_INDEX_FILE, metadata_path: Path = METADATA_FILE) -> bool:
        """
        Memuat indeks dan metadata dari disk.
        
        Mengembalikan:
            True jika berhasil dimuat, dan False jika sebaliknya.
        """
        if not index_path.exists() or not metadata_path.exists():
            print("Index files not found. Please build index first.")
            return False
            
        try:
            # Muat indeks FAISS
            loaded_index = faiss.read_index(str(index_path))
            print(f"Index loaded: {loaded_index.ntotal} vectors")
            
            # Muat metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                loaded_metadata = data['metadata']
                loaded_entries = [
                    TranslationEntry(**entry_dict) 
                    for entry_dict in data['entries']
                ]
            if loaded_index.ntotal != len(loaded_entries):
                raise ValueError(
                    "FAISS index and metadata contain different numbers of entries."
                )
            if len(loaded_metadata) != len(loaded_entries):
                raise ValueError(
                    "Metadata and entries contain different numbers of records."
                )

            self.index = loaded_index
            self.metadata = loaded_metadata
            self.entries = loaded_entries
            self.load_error = None
            print(f"Metadata loaded: {len(self.metadata)} entries")
            
            return True
            
        except Exception as e:
            self.load_error = e
            print(f"Error loading index: {e}")
            return False
            
    def search(self, query: str, top_k: int = 5) -> List[Tuple[TranslationEntry, float]]:
        """
        Mencari entri yang serupa.
        
        Parameter:
            query: Teks kueri dalam Bahasa Ocu.
            top_k: Jumlah hasil yang dikembalikan.
            
        Mengembalikan:
            Daftar tuple (TranslationEntry, skor_kemiripan).
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or build index first.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.index.ntotal != len(self.entries):
            raise ValueError("FAISS index and metadata are out of sync.")
        if self.index.ntotal == 0:
            return []
            
        # Ubah kueri menjadi penyematan vektor.
        query_embedding = np.asarray(
            self.embedding_model.encode_query(query),
            dtype=np.float32,
        ).reshape(1, -1)
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not match the FAISS index."
            )
        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding must contain only finite values.")
        
        # Jalankan pencarian.
        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )
        
        # Siapkan hasil pencarian.
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.entries):
                results.append((self.entries[idx], float(score)))
                
        return results
    

def build_vector_store(
    entries: Optional[Sequence[TranslationEntry]] = None,
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    embedding_model: Optional[EmbeddingModel] = None,
) -> VectorStore:
    """Membangun dan menyimpan basis data vektor dari dataset."""
    _validate_path_pair(index_path, metadata_path)
    selected_index_path = FAISS_INDEX_FILE if index_path is None else index_path
    selected_metadata_path = METADATA_FILE if metadata_path is None else metadata_path

    if entries is None:
        entries = DataLoader().load_all()
    
    # Bangun basis data vektor.
    vector_store = VectorStore(embedding_model=embedding_model)
    vector_store.build_index(entries)
    
    # Simpan ke disk.
    vector_store.save(selected_index_path, selected_metadata_path)
    
    return vector_store


def load_vector_store(
    index_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    build_if_missing: bool = True,
) -> VectorStore:
    """Memuat basis data vektor dari disk."""
    _validate_path_pair(index_path, metadata_path)
    selected_index_path = FAISS_INDEX_FILE if index_path is None else index_path
    selected_metadata_path = METADATA_FILE if metadata_path is None else metadata_path

    if not selected_index_path.exists() or not selected_metadata_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"Vector store files not found: {selected_index_path}; {selected_metadata_path}"
            )
        print("Building new vector store...")
        return build_vector_store(
            index_path=selected_index_path,
            metadata_path=selected_metadata_path,
            embedding_model=embedding_model,
        )

    vector_store = VectorStore(embedding_model=embedding_model)
    if vector_store.load(selected_index_path, selected_metadata_path):
        return vector_store

    if not build_if_missing:
        raise ValueError(
            f"Unable to load vector store files: {selected_index_path}; {selected_metadata_path}"
        )

    print("Building new vector store...")
    return build_vector_store(
        index_path=selected_index_path,
        metadata_path=selected_metadata_path,
        embedding_model=embedding_model,
    )


def _validate_path_pair(index_path: Optional[Path], metadata_path: Optional[Path]) -> None:
    """Mencegah artefak eksperimen dipasangkan dengan artefak bawaan."""
    if (index_path is None) != (metadata_path is None):
        raise ValueError("Provide both index_path and metadata_path, or omit both.")


if __name__ == "__main__":
    # Bangun dan uji basis data vektor.
    print("Building vector store...")
    store = build_vector_store()
    
    # Uji pencarian
    test_queries = [
        "deyen poi ka kampuong",
        "umah",
        "makan"
    ]
    
    print("\n" + "="*50)
    print("Testing search:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = store.search(query, top_k=3)
        for entry, score in results:
            print(f"  [{score:.3f}] {entry.bahasa_ocu} -> {entry.bahasa_indonesia} ({entry.entry_type})")
