"""
Tahap 4 - Retrieval Hybrid dan Penyusunan Konteks.
Menjalankan pencarian semantik dan leksikal, pemecahan kata,
lalu menyusun konteks referensi untuk proses penerjemahan.
"""
# Blok import: menyediakan tipe data dan komponen yang dipakai oleh retriever.
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from src.vector_store import VectorStore, load_vector_store
from src.data_loader import TranslationEntry, TextPreprocessor
from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, WORD_MATCH_THRESHOLD


# Blok struktur data: menyimpan entri dataset beserta skor hasil retrieval.
@dataclass
class RetrievalResult:
    """Hasil pengambilan informasi."""
    entry: TranslationEntry
    score: float

    # Blok pemformatan hasil: mengubah entri dan skor menjadi referensi untuk LLM.
    def to_context_string(self) -> str:
        """Memformat hasil sebagai string konteks untuk LLM."""
        import re
        type_labels = {
            'kosakata': 'Kosakata',
            'frasa': 'Frasa',
            'kalimat': 'Kalimat'
        }

        clean_ocu = re.sub(r'\s*\([^)]*\)', '', self.entry.bahasa_ocu).strip()
        clean_indo = re.sub(r'\s*\([^)]*\)', '', self.entry.bahasa_indonesia).strip()

        return (
            f"[{type_labels.get(self.entry.entry_type, self.entry.entry_type)}] "
            f"'{clean_ocu}' = '{clean_indo}'"
            f"{f' ({self.entry.keterangan})' if self.entry.keterangan else ''}"
        )


# Blok utama RAGRetriever: menjalankan pencarian, penyaringan, dan penyusunan konteks.
class RAGRetriever:
    """Pengambil informasi untuk penerjemahan berbasis RAG."""

    # Blok inisialisasi: menyiapkan vector store dan prapemroses teks.
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Menginisialisasi pengambil informasi dengan basis data vektor."""
        self.vector_store = vector_store or load_vector_store()
        self.preprocessor = TextPreprocessor()

    @staticmethod
    # Blok Levenshtein: menghitung jumlah minimum operasi edit antarteks.
    def _edit_distance(s1: str, s2: str) -> int:
        """Menghitung jarak edit Levenshtein antara dua string."""
        if len(s1) < len(s2):
            return RAGRetriever._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    @staticmethod
    # Blok normalisasi varian: memisahkan beberapa bentuk penulisan pada satu entri.
    def _normalize_entry_variants(entry_ocu: str) -> List[str]:
        """Memisahkan entri menjadi beberapa varian (mendukung format 'ANG / WAANG')."""
        variants = []
        if ' / ' in entry_ocu:
            for part in entry_ocu.split(' / '):
                variants.append(part.strip().lower())
        else:
            variants.append(entry_ocu.lower())
        return variants

    # Blok pemeriksaan leksikal: memvalidasi kecocokan persis, substring, dan typo.
    def _is_lexically_related(self, query_word: str, entry_ocu: str) -> bool:
        """
        Memeriksa apakah kata kueri berkaitan secara leksikal dengan suatu entri.
        Menggunakan kecocokan persis, kecocokan varian, substring (ketat), dan jarak edit.
        """
        query_lower = query_word.lower().strip()
        entry_variants = self._normalize_entry_variants(entry_ocu)

        for variant in entry_variants:
            if query_lower == variant:
                return True

            if len(query_lower) >= 3 and len(variant) >= 3:
                shorter_len = min(len(query_lower), len(variant))
                longer_len = max(len(query_lower), len(variant))
                length_ratio = shorter_len / longer_len

                if length_ratio >= 0.65:
                    if query_lower in variant or variant in query_lower:
                        return True
                else:
                    variant_words = variant.split()
                    if query_lower in variant_words:
                        return True
                    query_words = query_lower.split()
                    if variant in query_words:
                        return True

            max_dist = 0
            if len(query_lower) >= 7 or len(variant) >= 7:
                max_dist = 2
            elif len(query_lower) >= 4 or len(variant) >= 4:
                max_dist = 1

            if max_dist > 0:
                dist = self._edit_distance(query_lower, variant)
                longer_len = max(len(query_lower), len(variant))
                relative_dist = dist / longer_len if longer_len > 0 else 1.0
                if dist <= max_dist and relative_dist <= 0.25:
                    return True

        return False

    # Blok pemeriksaan relevansi: memastikan kandidat embedding juga berkaitan secara leksikal.
    def _relevance_check(self, query: str, entry_ocu: str, score: float) -> bool:
        """
        Memeriksa apakah entri yang diambil benar-benar relevan dengan kueri.
        SELALU memerlukan tumpang tindih leksikal; skor penyematan saja tidak dapat
        diandalkan untuk kata Ocu yang pendek.
        """
        query_words = set(query.lower().split())

        entry_variants = self._normalize_entry_variants(entry_ocu)
        all_entry_words = set()
        for v in entry_variants:
            all_entry_words.update(v.split())

        if query_words & all_entry_words:
            return True

        for qw in query_words:
            for ew in all_entry_words:
                if self._is_lexically_related(qw, ew):
                    return True
            for variant in entry_variants:
                if self._is_lexically_related(qw, variant):
                    return True

        if len(query_words) >= 3 and len(all_entry_words) >= 3:
            overlap = len(query_words & all_entry_words)
            if overlap >= 2:
                return True

        return False

    # Blok inferensi kata: mengambil padanan satu kata dari frasa dengan token yang sejajar.
    def _infer_single_word_mapping(self, word: str) -> Dict[str, str]:
        """Menyimpulkan kemungkinan terjemahan langsung satu kata dari entri tingkat frasa."""
        normalized_word = self.preprocessor.normalize_ocu(word)
        if not normalized_word:
            return {}

        for entry in self.vector_store.entries:
            if entry.entry_type not in {"frasa", "kosakata"}:
                continue

            source_text = self.preprocessor.normalize_ocu(entry.bahasa_ocu)
            target_text = self.preprocessor.normalize(entry.bahasa_indonesia)
            source_tokens = [tok for tok in source_text.split() if tok]
            target_tokens = [tok for tok in target_text.split() if tok]

            if not source_tokens or not target_tokens:
                continue
            if len(source_tokens) != len(target_tokens):
                continue
            if normalized_word not in source_tokens:
                continue

            idx = source_tokens.index(normalized_word)
            target_token = target_tokens[idx]
            if not target_token or len(target_token.split()) != 1:
                continue

            if len(target_token) <= 2:
                continue

            return {normalized_word: target_token}

        return {}

    # Blok retrieval utama: menormalisasi query, mencari kandidat, lalu menyaring hasil.
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        min_score: float = SIMILARITY_THRESHOLD
    ) -> List[RetrievalResult]:
        """
        Mengambil entri terjemahan yang relevan untuk suatu kueri.

        Parameter:
            query: Teks input dalam Bahasa Ocu.
            top_k: Jumlah maksimum hasil.
            min_score: Ambang batas minimum skor kemiripan.

        Mengembalikan:
            Daftar objek RetrievalResult.
        """
        query_normalized = self.preprocessor.normalize_ocu(query)

        search_results = self.vector_store.search(query_normalized, top_k=top_k * 3)

        results = []
        for entry, score in search_results:
            if score < min_score:
                continue

            entry_ocu_normalized = self.preprocessor.normalize_ocu(entry.bahasa_ocu)
            if self._relevance_check(query_normalized, entry_ocu_normalized, score):
                results.append(RetrievalResult(entry=entry, score=score))

        return results[:top_k]

    # Blok retrieval terurai: mencari query utuh, frasa, dan kata satuan sesuai prioritas.
    def retrieve_with_word_breakdown(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Mengambil konteks untuk kueri utuh, frasa, dan kata satuan.

        Urutan prioritas:
        1. Kecocokan kueri utuh.
        2. Kecocokan frasa multi-kata (bigram, trigram).
        3. Kecocokan kata satuan.

        Parameter:
            query: Teks input dalam Bahasa Ocu.
            top_k: Jumlah maksimum hasil untuk setiap pencarian.

        Mengembalikan:
            Kamus berisi hasil 'full_query' dan 'word_breakdown'.
        """
        results = {
            'full_query': [],
            'word_breakdown': {}
        }

        results['full_query'] = self.retrieve(query, top_k=top_k)

        words = query.lower().split()

        if len(words) == 1:
            word = words[0]
            if len(word) > 1:
                direct_matches = self._text_search(word, top_k=6)
                inferred_mapping = self._infer_single_word_mapping(word)

                if inferred_mapping:
                    inferred_entry = TranslationEntry(
                        id=f"fallback-{word}",
                        bahasa_ocu=word,
                        bahasa_indonesia=inferred_mapping[word],
                        keterangan="Fallback kata tunggal",
                        entry_type="kosakata",
                    )
                    results['word_breakdown'][word] = [RetrievalResult(entry=inferred_entry, score=1.0)]
                    return results

                if direct_matches:
                    direct_matches = [m for m in direct_matches if m.entry.entry_type == 'kosakata'] or direct_matches
                    results['word_breakdown'][word] = direct_matches[:2]
                return results

        covered_positions = set()

        for n in [3, 2]:
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                positions = set(range(i, i + n))
                if positions & covered_positions:
                    continue

                phrase = ' '.join(words[i:i+n])

                phrase_matches = self._phrase_search(phrase, top_k=2)
                if phrase_matches:
                    results['word_breakdown'][phrase] = phrase_matches
                    covered_positions |= positions

        for i, word in enumerate(words):
            if i in covered_positions:
                continue
            if len(word) <= 1:
                continue

            text_candidates = self._text_search(word, top_k=6)
            exact_vocabulary = []
            for candidate in text_candidates:
                if candidate.entry.entry_type != 'kosakata':
                    continue
                variants = self._normalize_entry_variants(candidate.entry.bahasa_ocu)
                if word.lower().strip() in variants:
                    exact_vocabulary.append(candidate)

            if exact_vocabulary:
                results['word_breakdown'][word] = exact_vocabulary[:2]
                continue

            inferred_mapping = self._infer_single_word_mapping(word)
            if inferred_mapping:
                inferred_entry = TranslationEntry(
                    id=f"fallback-{word}",
                    bahasa_ocu=word,
                    bahasa_indonesia=inferred_mapping[word],
                    keterangan="Inferensi token dari frasa sejajar",
                    entry_type="kosakata",
                )
                results['word_breakdown'][word] = [
                    RetrievalResult(entry=inferred_entry, score=1.0)
                ]
                continue

            word_results = self.retrieve(word, top_k=2, min_score=WORD_MATCH_THRESHOLD)
            if word_results:
                safe_word_results = [
                    result for result in word_results
                    if len(self.preprocessor.normalize_ocu(result.entry.bahasa_ocu).split()) == 1
                    and self._is_lexically_related(word, result.entry.bahasa_ocu)
                ]
                if safe_word_results:
                    results['word_breakdown'][word] = safe_word_results
                    continue

            text_matches = [
                match for match in text_candidates
                if len(self.preprocessor.normalize_ocu(match.entry.bahasa_ocu).split()) == 1
            ][:2]
            if text_matches:
                results['word_breakdown'][word] = text_matches

        return results

    # Blok pencarian frasa: mencocokkan frasa pendek secara langsung pada dataset.
    def _phrase_search(self, phrase: str, top_k: int = 2) -> List[RetrievalResult]:
        """
        Mencari frasa multi-kata dalam dataset.
        Hanya mencocokkan entri kosakata dan frasa (BUKAN kalimat penuh).
        Jumlah kata entri harus mendekati jumlah kata frasa.
        """
        phrase_lower = phrase.lower().strip()
        phrase_words = set(phrase_lower.split())
        phrase_word_count = len(phrase_lower.split())
        matches = []

        for entry in self.vector_store.entries:
            if entry.entry_type == 'kalimat':
                continue

            entry_ocu = entry.bahasa_ocu.lower().strip()
            entry_words = set(entry_ocu.split())
            entry_word_count = len(entry_ocu.split())

            if abs(entry_word_count - phrase_word_count) > 1:
                continue

            if phrase_lower == entry_ocu:
                matches.append(RetrievalResult(entry=entry, score=1.0))
                continue

            variants = self._normalize_entry_variants(entry_ocu)
            if phrase_lower in variants:
                matches.append(RetrievalResult(entry=entry, score=1.0))
                continue

            if phrase_words == entry_words and len(phrase_words) >= 2:
                matches.append(RetrievalResult(entry=entry, score=0.95))
                continue

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]

    # Blok pencarian teks: menjadi fallback leksikal ketika pencarian embedding belum memadai.
    def _text_search(self, word: str, top_k: int = 2) -> List[RetrievalResult]:
        """
        Pencarian berbasis teks sebagai cadangan ketika penyematan vektor gagal.
        Memindai seluruh entri untuk mencari kecocokan leksikal.
        """
        matches = []
        word_lower = word.lower().strip()

        for entry in self.vector_store.entries:
            entry_ocu = entry.bahasa_ocu.lower()

            if self._is_lexically_related(word_lower, entry_ocu):
                variants = self._normalize_entry_variants(entry_ocu)
                if word_lower in variants:
                    score = 1.0
                elif any(word_lower in v for v in variants):
                    score = 0.9
                else:
                    score = 0.75

                matches.append(RetrievalResult(entry=entry, score=score))

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]

    # Blok validasi kalimat: membedakan kecocokan persis dan kecocokan yang hanya mirip.
    def _check_sentence_exact_match(self, query: str, entry_ocu: str) -> bool:
        """
        Memeriksa apakah entri kalimat cocok persis dengan kueri (kata demi kata).
        Digunakan untuk menentukan apakah referensi kalimat dapat dipercaya apa adanya.
        """
        q_words = set(query.lower().split())
        e_words = set(entry_ocu.lower().split())

        if q_words == e_words:
            return True

        if q_words and e_words:
            intersection = q_words & e_words
            q_ratio = len(intersection) / len(q_words)
            e_ratio = len(intersection) / len(e_words)
            if q_ratio >= 0.9 and e_ratio >= 0.9:
                return True

        return False

    # Blok penyusunan konteks: menggabungkan hasil retrieval menjadi referensi prompt LLM.
    def format_context_for_llm(
        self,
        retrieval_results: List[RetrievalResult],
        word_breakdown: Optional[Dict[str, List[RetrievalResult]]] = None,
        original_query: str = ""
    ) -> str:
        """
        Memformat hasil pengambilan informasi sebagai string konteks untuk LLM.

        Parameter:
            retrieval_results: Hasil dari retrieve().
            word_breakdown: Hasil pemecahan kata per kata (opsional).
            original_query: Kueri input asli untuk perbandingan.

        Mengembalikan:
            String konteks yang sudah diformat.
        """
        context_parts = []

        import re
        if word_breakdown:
            context_parts.append("=== ARTI KATA PER KATA (PRIORITAS UTAMA) ===")
            for word, results in word_breakdown.items():
                if results:
                    best = results[0]
                    clean_indo = re.sub(r'\s*\([^)]*\)', '', best.entry.bahasa_indonesia).strip()
                    context_parts.append(
                        f"  - '{word}' = '{clean_indo}'"
                    )

        if retrieval_results:
            has_similar_but_not_exact = False
            context_parts.append("\nReferensi terjemahan:")
            for result in retrieval_results:
                entry_ocu = result.entry.bahasa_ocu
                is_sentence = result.entry.entry_type == 'kalimat'

                if is_sentence and original_query:
                    if self._check_sentence_exact_match(original_query, entry_ocu):
                        context_parts.append(f"  - [COCOK PERSIS] {result.to_context_string()}")
                    else:
                        has_similar_but_not_exact = True
                        context_parts.append(f"  - [MIRIP, BUKAN INPUT ANDA] {result.to_context_string()}")
                else:
                    context_parts.append(f"  - {result.to_context_string()}")

            if has_similar_but_not_exact:
                context_parts.append(
                    "\nâš ï¸ PERINGATAN: Kalimat referensi di atas MIRIP tapi BUKAN SAMA PERSIS dengan input."
                    "\nJANGAN SALIN terjemahan kalimat tersebut! Gunakan ARTI KATA PER KATA di atas untuk menyusun terjemahan baru."
                )

        return "\n".join(context_parts) if context_parts else "Tidak ada referensi yang ditemukan."
