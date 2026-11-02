"""
Tahap 9 - Penyajian Layanan API.
Menyediakan endpoint REST untuk terjemahan, statistik, dan penyajian
antarmuka web.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys

# Tambahkan direktori induk ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translator import OcuTranslator
from src.vector_store import VectorStore, load_vector_store

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Ocu Translator API",
    description="API untuk menerjemahkan Bahasa Ocu ke Bahasa Indonesia menggunakan RAG",
    version="1.0.0"
)

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instans penerjemah global (dimuat saat diperlukan).
translator: Optional[OcuTranslator] = None
vector_store: Optional[VectorStore] = None


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    success: bool
    input_text: str
    translation: str
    error: Optional[str] = None


def get_translator() -> OcuTranslator:
    """Mengambil atau menginisialisasi penerjemah."""
    global translator
    if translator is None:
        print("Initializing translator...")
        translator = OcuTranslator(vector_store=get_vector_store())
    return translator


def get_vector_store() -> VectorStore:
    """Mengambil atau menginisialisasi basis pengetahuan penerjemah."""
    global vector_store
    if vector_store is None:
        print("Initializing vector store...")
        vector_store = load_vector_store()
    return vector_store


@app.get("/")
async def root():
    """Menyajikan halaman HTML utama."""
    return FileResponse(
        "frontend/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Menerjemahkan Bahasa Ocu ke Bahasa Indonesia."""
    try:
        trans = get_translator()
        result = trans.translate(text=request.text)
        
        return TranslateResponse(
            success=result.success,
            input_text=result.input_text,
            translation=result.translated_text,
            error=result.error_message
        )
    except Exception as e:
        return TranslateResponse(
            success=False,
            input_text=request.text,
            translation="",
            error=str(e)
        )


@app.get("/api/health")
async def health_check():
    """Endpoint pemeriksaan kesehatan layanan."""
    return {"status": "healthy", "service": "ocu-translator"}


@app.get("/api/stats")
async def get_stats():
    """Mengambil statistik basis data."""
    try:
        store = get_vector_store()
        
        # Hitung berdasarkan tipe.
        kosakata = len([e for e in store.entries if e.entry_type == 'kosakata'])
        frasa = len([e for e in store.entries if e.entry_type == 'frasa'])
        kalimat = len([e for e in store.entries if e.entry_type == 'kalimat'])
        
        return {
            "total_entries": len(store.entries),
            "kosakata": kosakata,
            "frasa": frasa,
            "kalimat": kalimat
        }
    except Exception as e:
        return {"error": str(e)}


# Pasang berkas statis.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Ocu Translator API Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
