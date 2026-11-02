"""
Modul konfigurasi untuk sistem penerjemahan RAG.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Memuat variabel lingkungan
load_dotenv()

# Path dasar
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Berkas dataset
KOSAKATA_FILE = DATASET_DIR / "kosakata_bahasa_ocu.csv"
FRASA_FILE = DATASET_DIR / "frasa_bahasa_ocu.csv"
KALIMAT_FILE = DATASET_DIR / "kalimat_bahasa_ocu.csv"

# Konfigurasi penyematan vektor (embedding).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384  # Dimensi untuk model yang dipilih

# Konfigurasi FAISS
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "ocu_faiss.index"
METADATA_FILE = VECTOR_STORE_DIR / "ocu_metadata.pkl"

# Konfigurasi pengambilan informasi (retrieval).
TOP_K_RESULTS = 3  # Jumlah hasil yang diambil
SIMILARITY_THRESHOLD = 0.5  # Batas minimum skor kemiripan untuk retrieval
WORD_MATCH_THRESHOLD = 0.6  # Batas minimum skor untuk kecocokan kata tunggal
EXACT_MATCH_THRESHOLD = 0.95  # Batas skor sebelum verifikasi exact match
LOW_CONFIDENCE_THRESHOLD = 0.7  # Di bawah nilai ini, tambahkan peringatan pada prompt LLM.

# Konfigurasi LLM (API Ollama lokal)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_ENDPOINT = os.getenv("OLLAMA_GENERATE_ENDPOINT", "/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "translategemma:4b"))

# Parameter LLM
LLM_TEMPERATURE = 0.1  # Temperatur sangat rendah untuk terjemahan deterministik
LLM_MAX_TOKENS = 1024

# Pastikan direktori tersedia
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
