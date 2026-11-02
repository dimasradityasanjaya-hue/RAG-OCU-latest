# RAG Translation System - Bahasa Ocu ke Bahasa Indonesia

Sistem penerjemahan Bahasa Ocu (Bahasa Kampar) ke Bahasa Indonesia menggunakan **Retrieval-Augmented Generation (RAG)** dengan **Ollama (Gemma) LLM lokal**.

## 🎯 Fitur

- ✅ Penerjemahan kosakata, frasa, dan kalimat Bahasa Ocu
- ✅ RAG-based translation untuk akurasi tinggi
- ✅ Vector database (FAISS) untuk retrieval konteks terjemahan
- ✅ Integrasi dengan Ollama (Gemma)
- ✅ CLI interaktif untuk penggunaan mudah

## 📁 Struktur Proyek

```
RAG+OCU/
├── dataset/
│   ├── kosakata_bahasa_ocu.csv
│   ├── frasa_bahasa_ocu.csv
│   └── kalimat_bahasa_ocu.csv
├── src/
│   ├── __init__.py
│   ├── config.py          # Konfigurasi sistem
│   ├── data_loader.py     # Loading dan preprocessing dataset
│   ├── embedding.py       # Model embedding (sentence-transformers)
│   ├── vector_store.py    # FAISS vector database
│   ├── retriever.py       # RAG retriever
│   ├── llm.py             # Integrasi Ollama Local API
│   ├── translator.py      # Main translator class
│   └── cli.py             # Command-line interface
├── scripts/
│   └── build_index.py     # Script untuk build vector index
├── vector_store/          # Folder untuk menyimpan FAISS index
├── .env.example           # Template environment variables
├── requirements.txt       # Dependencies
└── README.md
```

## 🚀 Instalasi

### 1. Clone Repository

```bash
cd "D:\TUGAS AKHIR\RAG+OCU"
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

```bash
copy .env.example .env
```

Edit file `.env` dan sesuaikan konfigurasi Ollama:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATE_ENDPOINT=/api/generate
OLLAMA_MODEL=gemma:2b
LLM_MODEL=gemma:2b
```

Pastikan model sudah ditarik di mesin lokal:

```bash
ollama pull gemma:2b
```

### 5. Build Vector Index

```bash
python scripts/build_index.py
```

## 💻 Penggunaan

### Mode Interaktif

```bash
python -m src.cli
```

atau

```bash
python -m src.cli --interactive
```

### Terjemahkan Satu Kalimat

```bash
python -m src.cli "deyen poi ka kampuong"
```

### Rebuild Vector Index

```bash
python -m src.cli --rebuild
```

### Evaluasi Ocu ke Indonesia

```bash
python scripts/run_evaluation.py
```

Output evaluasi akan disimpan ke folder `evaluation_results/` dalam format JSON, TXT, dan CSV.

## 📖 Contoh Penggunaan

```
🔤 Masukkan teks Bahasa Ocu: deyen poi ka kampuong

⏳ Menerjemahkan...

📝 Terjemahan:
   Saya pergi ke kampung
```

## 🔧 Arsitektur Sistem

```
Input Bahasa Ocu
      ↓
Preprocessing (normalisasi teks)
      ↓
Retrieval (Vector Database - FAISS)
      ↓
Prompt Augmentation (menambahkan referensi RAG)
      ↓
LLM Ollama (Gemma)
      ↓
Output Terjemahan Bahasa Indonesia
```

## 📊 Dataset

| Dataset | Jumlah | Deskripsi |
|---------|--------|-----------|
| Kosakata | 735 | Kata-kata dasar Bahasa Ocu |
| Frasa | 88 | Frasa dan ungkapan |
| Kalimat | 50 | Kalimat lengkap |

## ⚙️ Konfigurasi

Konfigurasi dapat diubah di `src/config.py`:

- `EMBEDDING_MODEL`: Model embedding (default: paraphrase-multilingual-MiniLM-L12-v2)
- `TOP_K_RESULTS`: Jumlah hasil retrieval (default: 5)
- `LLM_TEMPERATURE`: Temperature LLM (default: 0.3)

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademis (Tugas Akhir).

## 👤 Author

Pengembang Sistem RAG Bahasa Ocu

## Eksperimen Holdout 85:15 Terisolasi

Indeks penuh yang digunakan proyek saat ini berisi 873 entri. Untuk evaluasi
holdout yang terisolasi, data dibagi secara deterministik menjadi 748 entri
latih/FAISS dan 125 entri uji: kosakata 635/100, frasa 73/15, dan kalimat
40/10.

Holdout berarti sebuah entri Ocu yang sudah dinormalisasi secara persis tidak
boleh muncul di kedua split. Kata-kata penyusunnya masih dapat muncul pada
entri latih yang lain; batasannya adalah pada entri lengkap, bukan pada setiap
kata penyusun. Semua dataset, vector store, dan hasil evaluasi lama tetap tidak
disentuh.

Setiap eksperimen menulis keluaran secara terisolasi pada
`experiments/<nama-eksperimen>/`:

```
experiments/<nama-eksperimen>/
  dataset/
    train_kosakata.csv
    train_frasa.csv
    train_kalimat.csv
    test_set.csv
    split_manifest.json
  vector_store/
    ocu_faiss.index
    ocu_metadata.pkl
  evaluation_results/  # hanya pada evaluasi penuh
```

`dataset/split_manifest.json` menyimpan bukti split, termasuk jumlah per tipe,
ID train/test, jumlah overlap ternormalisasi, checksum sumber dan artefak
terlindungi, serta hash indeks dan metadata FAISS.

Jalankan dari environment proyek:

```powershell
.\.venv_holdout\Scripts\python.exe scripts\run_holdout_experiment.py --experiment-name holdout_85_15_smoke --prepare-only
.\.venv_holdout\Scripts\python.exe scripts\run_holdout_experiment.py --experiment-name holdout_85_15_v1
```

Opsi `--prepare-only` hanya membuat split dan indeks FAISS; opsi ini tidak
menjalankan translator maupun evaluator. Direktori eksperimen yang sudah ada
selalu ditolak untuk mencegah penimpaan, sehingga eksekusi ulang harus memakai
nama eksperimen baru. Evaluasi penuh membutuhkan Ollama yang sudah
dikonfigurasi dan aktif.
