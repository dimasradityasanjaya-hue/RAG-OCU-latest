"""
Tahap 1 - Persiapan Data dan Normalisasi Basis Pengetahuan.
Memuat dataset kosakata, frasa, dan kalimat serta menstandarkan teks
sebelum masuk ke proses embedding dan retrieval.
"""
import pandas as pd
import re
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

from src.config import KOSAKATA_FILE, FRASA_FILE, KALIMAT_FILE


@dataclass
class TranslationEntry:
    """Kelas data untuk satu entri terjemahan."""
    id: str
    bahasa_ocu: str
    bahasa_indonesia: str
    keterangan: str
    entry_type: str  # 'kosakata', 'frasa', 'kalimat'
    
    def to_dict(self) -> Dict:
        """Mengubah entri menjadi kamus untuk penyimpanan metadata."""
        return {
            'id': self.id,
            'bahasa_ocu': self.bahasa_ocu,
            'bahasa_indonesia': self.bahasa_indonesia,
            'keterangan': self.keterangan,
            'entry_type': self.entry_type
        }


class TextPreprocessor:
    """Prapemroses untuk normalisasi teks."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalisasi teks: ubah ke huruf kecil, hapus spasi berlebih"""
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        # Hapus spasi berlebih.
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def normalize_ocu(text: str) -> str:
        """Normalisasi teks Bahasa Ocu"""
        text = TextPreprocessor.normalize(text)
        # Normalisasi khusus Bahasa Ocu dapat ditambahkan di sini.
        return text


class DataLoader:
    """Pemuat dataset Bahasa Ocu."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.entries: List[TranslationEntry] = []

    def _read_csv_with_fallbacks(self, filepath: Path, sep: str) -> pd.DataFrame:
        """Membaca CSV dengan beberapa pengodean yang umum pada berkas Windows."""
        encodings = ('utf-8-sig', 'utf-8', 'cp1252', 'latin1')
        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(filepath, sep=sep, encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error

        if last_error is not None:
            raise last_error

        return pd.read_csv(filepath, sep=sep)
        
    def _load_csv(self, filepath: Path, entry_type: str, indonesian_col: str, ocu_col: str = 'Bahasa Ocu') -> List[TranslationEntry]:
        """Memuat satu berkas CSV dan mengembalikan daftar entri."""
        entries = []
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return entries
            
        try:
            # Coba pemisah titik koma terlebih dahulu, lalu koma.
            try:
                df = self._read_csv_with_fallbacks(filepath, sep=';')
                # Periksa apakah kolom tersedia
                if ocu_col not in df.columns:
                    df = self._read_csv_with_fallbacks(filepath, sep=',')
            except:
                df = self._read_csv_with_fallbacks(filepath, sep=',')
            
            # Normalisasi nama kolom (tidak sensitif huruf besar/kecil)
            df.columns = [col.strip() for col in df.columns]
            
            for _, row in df.iterrows():
                # Coba variasi nama kolom yang berbeda
                id_val = row.get('Id', row.get('id', row.get('ID', '')))
                ocu_val = row.get('Bahasa Ocu', row.get('bahasa_ocu', row.get('BAHASA OCU', '')))
                indo_val = row.get(indonesian_col, row.get(indonesian_col.lower(), row.get(indonesian_col.replace('_', ' ').title(), '')))
                ket_val = row.get('Keterangan', row.get('keterangan', row.get('KETERANGAN', '')))
                # --- BERSIHKAN & NORMALISASI ---
                # Gunakan TextPreprocessor untuk membuat huruf kecil, menghapus spasi ekstra,
                # dan menormalisasi teks
                ocu_clean = self.preprocessor.normalize_ocu(ocu_val if pd.notna(ocu_val) else '')
                indo_clean = self.preprocessor.normalize(indo_val if pd.notna(indo_val) else '')
                ket_clean = str(ket_val).strip() if pd.notna(ket_val) else ''

                # --- MODELIKAN BARIS MENJADI ENTRI ---
                # Buat instans TranslationEntry yang merepresentasikan
                # satu unit data dalam basis pengetahuan (kosakata/frasa/kalimat).
                entry = TranslationEntry(
                    id=str(id_val) if pd.notna(id_val) else '',
                    bahasa_ocu=ocu_clean,
                    bahasa_indonesia=indo_clean,
                    keterangan=ket_clean,
                    entry_type=entry_type
                )
                
                # Lewati entri kosong
                if entry.bahasa_ocu and entry.bahasa_indonesia:
                    entries.append(entry)
                    
            print(f"Loaded {len(entries)} entries from {filepath.name}")
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            
        return entries
    
    def load_all(self) -> List[TranslationEntry]:
        """Memuat seluruh dataset."""
        self.entries = []
        
        # Muat kosakata — setiap baris menjadi satu TranslationEntry.
        kosakata = self._load_csv(KOSAKATA_FILE, 'kosakata', 'Terjemahan Indonesia')
        self.entries.extend(kosakata)
        
        # Muat frasa — setiap baris menjadi satu TranslationEntry.
        frasa = self._load_csv(FRASA_FILE, 'frasa', 'Frasa Indonesia')
        self.entries.extend(frasa)
        
        # Muat kalimat — setiap baris menjadi satu TranslationEntry.
        kalimat = self._load_csv(KALIMAT_FILE, 'kalimat', 'Terjemahan Indonesia')
        self.entries.extend(kalimat)
        
        print(f"\nTotal entries loaded: {len(self.entries)}")
        print(f"  - Kosakata: {len(kosakata)}")
        print(f"  - Frasa: {len(frasa)}")
        print(f"  - Kalimat: {len(kalimat)}")
        
        return self.entries
    

if __name__ == "__main__":
    # Uji pemuatan data
    loader = DataLoader()
    entries = loader.load_all()
    
    print("\nSample entries:")
    for entry in entries[:5]:
        print(f"  {entry.entry_type}: {entry.bahasa_ocu} -> {entry.bahasa_indonesia}")
