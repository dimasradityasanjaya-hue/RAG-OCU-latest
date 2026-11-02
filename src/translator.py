"""
Tahap 5 - Orkestrasi Pipeline RAGTrans.
Mengatur input, preprocessing, retrieval, penyusunan konteks,
keputusan exact match atau fallback, pemanggilan LLM, dan hasil akhir.
"""
from typing import Optional, Dict
from dataclasses import dataclass

from src.retriever import RAGRetriever
from src.llm import LLMTranslator
from src.vector_store import VectorStore, load_vector_store, build_vector_store
from src.data_loader import TextPreprocessor
from src import config


@dataclass
class TranslationResult:
    """Hasil terjemahan lengkap."""
    input_text: str
    translated_text: str
    context_used: str
    retrieval_results: list
    success: bool
    error_message: Optional[str] = None
    
class OcuTranslator:
    """
    Kelas utama penerjemah Bahasa Ocu ke Bahasa Indonesia.
    Menggabungkan pengambilan informasi RAG dengan penerjemahan LLM.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rebuild_index: bool = False,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Menginisialisasi penerjemah.
        
        Parameter:
            api_key: Parameter kompatibilitas yang sudah usang (tidak digunakan Ollama).
            rebuild_index: Menentukan apakah indeks vektor dibangun ulang dari awal.
            vector_store: Basis data vektor yang sudah dibangun untuk digunakan
                tanpa memuat atau membangunnya ulang.
        """
        print("Initializing Ocu Translator...")
        
        # Inisialisasi basis data vektor.
        if vector_store is not None:
            print("Using supplied vector store...")
        elif rebuild_index:
            print("Building vector store from dataset...")
            vector_store = build_vector_store()
        else:
            print("Loading vector store...")
            vector_store = load_vector_store()
        
        # Inisialisasi pengambil informasi.
        self.retriever = RAGRetriever(vector_store)
        
        # Inisialisasi LLM.
        print("Initializing LLM translator...")
        self.llm = LLMTranslator(api_key)
        
        # Prapemroses untuk normalisasi masukan (input).
        self.preprocessor = TextPreprocessor()
        self._ambiguity_cache: Optional[Dict[str, bool]] = None
        
        print("Translator ready!")

    def _clean_translation_text(self, text: str) -> str:
        import re
        return re.sub(r'\s*\([^)]*\)', '', text).strip()

    def _split_translation_variants(self, text: str) -> list:
        import re
        variants = [self._clean_translation_text(part) for part in re.split(r'\s*/\s*', text)]
        return [variant for variant in variants if variant]

    def _build_ambiguity_cache(self) -> Dict[str, bool]:
        meanings_by_source: Dict[str, set] = {}

        for entry in getattr(self.retriever.vector_store, 'entries', []):
            # Aturan mempertahankan teks asli hanya berlaku untuk kosakata.
            if entry.entry_type != 'kosakata':
                continue

            source_variants = {
                self.preprocessor.normalize_ocu(variant)
                for variant in entry.bahasa_ocu.split('/')
                if variant.strip()
            }
            translation_variants = {
                self.preprocessor.normalize(variant)
                for variant in self._split_translation_variants(entry.bahasa_indonesia)
                if variant.strip()
            }

            for source_variant in source_variants:
                meanings_by_source.setdefault(source_variant, set()).update(translation_variants)

        return {
            source: len(meanings) > 1
            for source, meanings in meanings_by_source.items()
        }

    def _has_ambiguous_exact_source(self, query_text: str) -> bool:
        normalized_query = self.preprocessor.normalize_ocu(query_text)
        if self._ambiguity_cache is None:
            self._ambiguity_cache = self._build_ambiguity_cache()
        return self._ambiguity_cache.get(normalized_query, False)

    def _should_preserve_original_for_exact_match(self, query_text: str, entry) -> bool:
        if entry.entry_type == 'kalimat':
            return False

        normalized_query = self.preprocessor.normalize_ocu(query_text)
        if self._ambiguity_cache is None:
            self._ambiguity_cache = self._build_ambiguity_cache()

        return self._ambiguity_cache.get(normalized_query, False)
        
    def translate(
        self,
        text: str,
        include_explanation: bool = False,
    ) -> TranslationResult:
        """Menerjemahkan teks Bahasa Ocu ke Bahasa Indonesia."""
        try:
            normalized_text = self.preprocessor.normalize_ocu(text)

            if not include_explanation and self._has_ambiguous_exact_source(normalized_text):
                return TranslationResult(
                    input_text=text,
                    translated_text=text.strip(),
                    context_used="",
                    retrieval_results=[],
                    success=True
                )

            retrieval_data = self.retriever.retrieve_with_word_breakdown(normalized_text)
            full_results = retrieval_data['full_query']
            word_breakdown = retrieval_data['word_breakdown']
            best_score = full_results[0].score if full_results else 0.0
            has_word_matches = len(word_breakdown) > 0

            if not full_results and not has_word_matches:
                context = (
                    "PERINGATAN: TIDAK ADA referensi yang ditemukan untuk input ini. "
                    "Terjemahkan hanya kata-kata yang Anda yakin, dan pertahankan "
                    "kata-kata yang tidak diketahui apa adanya. "
                    "JANGAN mengarang terjemahan."
                )
            elif best_score < config.LOW_CONFIDENCE_THRESHOLD and not has_word_matches:
                context = (
                    "PERINGATAN: Referensi berikut memiliki tingkat kemiripan RENDAH. "
                    "Hanya gunakan jika benar-benar relevan dengan input.\n\n"
                )
                context += self.retriever.format_context_for_llm(
                    retrieval_results=full_results,
                    word_breakdown=word_breakdown,
                    original_query=normalized_text
                )
            else:
                context = self.retriever.format_context_for_llm(
                    retrieval_results=full_results,
                    word_breakdown=word_breakdown,
                    original_query=normalized_text
                )

            exact_match = self._check_exact_match(retrieval_data, normalized_text)
            if exact_match and not include_explanation:
                best_result = full_results[0] if full_results else None
                is_kalimat = best_result and best_result.entry.entry_type == 'kalimat'
                if best_result and best_result.entry.entry_type in ('kosakata', 'frasa'):
                    if self._should_preserve_original_for_exact_match(normalized_text, best_result.entry):
                        return TranslationResult(
                            input_text=text,
                            translated_text=text.strip(),
                            context_used=context,
                            retrieval_results=full_results,
                            success=True
                        )
                if best_result and self._should_preserve_original_for_exact_match(normalized_text, best_result.entry):
                    return TranslationResult(
                        input_text=text,
                        translated_text=text.strip(),
                        context_used=context,
                        retrieval_results=full_results,
                        success=True
                    )
                if is_kalimat and self.llm._needs_spok_validation(exact_match):
                    validated = self.llm._validate_spok(normalized_text, exact_match)
                else:
                    validated = self.llm._format_output(exact_match, False)
                return TranslationResult(
                    input_text=text,
                    translated_text=validated,
                    context_used=context,
                    retrieval_results=full_results,
                    success=True
                )

            translated = self.llm.translate(
                text=normalized_text,
                context=context,
                include_explanation=include_explanation,
            )

            return TranslationResult(
                input_text=text,
                translated_text=translated,
                context_used=context,
                retrieval_results=full_results,
                success=True
            )

        except Exception as e:
            return TranslationResult(
                input_text=text,
                translated_text="",
                context_used="",
                retrieval_results=[],
                success=False,
                error_message=str(e)
            )
    
    def _check_exact_match(self, retrieval_data: Dict, query_text: str = "") -> Optional[str]:
        """Memeriksa apakah ada kecocokan persis yang BENAR-BENAR nyata.
        
        Menggunakan perbandingan teks, bukan hanya skor penyematan vektor.
        Skor penyematan dapat menyesatkan untuk kalimat yang mirip tetapi berbeda.
        """
        import re
        full_results = retrieval_data.get('full_query', [])
        if not full_results:
            return None
            
        best_result = full_results[0]
        
        if best_result.score >= config.EXACT_MATCH_THRESHOLD and query_text:
            # Bandingkan kata aktual, bukan hanya skor penyematan vektor.
            query_words = set(query_text.lower().split())
            entry_words = set(best_result.entry.bahasa_ocu.lower().split())
            
            # Fungsi bantu untuk menghapus teks dalam tanda kurung.
            def clean_result(text):
                return re.sub(r'\s*\([^)]*\)', '', text).strip()
            
            # Untuk kosakata/frasa: cek varian
            if best_result.entry.entry_type in ('kosakata', 'frasa'):
                entry_variants = [v.strip().lower() for v in best_result.entry.bahasa_ocu.split('/')]
                if query_text.lower().strip() in entry_variants:
                    return clean_result(best_result.entry.bahasa_indonesia)
            
            # Untuk kalimat: harus cocok SEMUA kata di kedua arah
            if best_result.entry.entry_type == 'kalimat':
                if query_words == entry_words:
                    return clean_result(best_result.entry.bahasa_indonesia)
                # Jika kata tidak cocok persis, jangan dikembalikan - biarkan LLM menyusun
                return None
            
            # Untuk tipe lain dengan kecocokan kata persis
            if query_words == entry_words:
                return clean_result(best_result.entry.bahasa_indonesia)
        
        return None
    
def create_translator(
    api_key: Optional[str] = None,
    rebuild_index: bool = False,
    vector_store: Optional[VectorStore] = None,
) -> OcuTranslator:
    """
    Fungsi pabrik untuk membuat instans penerjemah.
    
    Parameter:
        api_key: Parameter kompatibilitas yang sudah usang.
        rebuild_index: Menentukan apakah indeks vektor dibangun ulang.
        vector_store: Basis data vektor yang sudah dibangun untuk digunakan tanpa
            memuat atau membangunnya ulang.
        
    Mengembalikan:
        Instans OcuTranslator.
    """
    return OcuTranslator(
        api_key=api_key,
        rebuild_index=rebuild_index,
        vector_store=vector_store,
    )


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("OCU TRANSLATOR - Bahasa Ocu ke Bahasa Indonesia")
    print("="*60)
    
    # Cek apakah indeks perlu dibangun ulang
    rebuild = "--rebuild" in sys.argv
    
    try:
        # Buat penerjemah.
        translator = create_translator(rebuild_index=rebuild)
        
        # Uji terjemahan
        test_sentences = [
            "deyen poi ka kampuong",
            "inyo sedang manyungka di umah",
            "waang olah makan",
            "kami beajau besamo di sikolah"
        ]
        
        print("\n" + "="*60)
        print("TEST TERJEMAHAN")
        print("="*60)
        
        for sentence in test_sentences:
            print(f"\nInput: {sentence}")
            result = translator.translate(sentence)
            
            if result.success:
                print(f"Terjemahan: {result.translated_text}")
            else:
                print(f"Error: {result.error_message}")
            print("-" * 40)
            
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nUntuk menggunakan sistem ini, silakan:")
        print("1. Buat file .env dari .env.example")
        print("2. Pastikan Ollama berjalan di endpoint yang dikonfigurasi")
