"""
Tahap 6 - Prompt Augmentation, Generasi, dan Post-processing.
Memasukkan konteks retrieval ke prompt, memanggil model Ollama,
menerapkan fallback, serta merapikan dan memvalidasi keluaran terjemahan.
"""
from typing import Optional, Set
import json
import re
from urllib import request, error
from urllib.parse import urljoin

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_GENERATE_ENDPOINT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)


# Prompt sistem untuk penerjemahan Bahasa Ocu ke Bahasa Indonesia.
SYSTEM_PROMPT_OCU_TO_ID = """Anda adalah penerjemah Bahasa Ocu ke Bahasa Indonesia.

Aturan utama:
1. Gunakan referensi RAG sebagai sumber utama arti kata.
2. Jika referensi bertanda [MIRIP, BUKAN INPUT ANDA], jangan salin terjemahan kalimatnya.
3. Susun hasil akhir menjadi kalimat Indonesia alami dengan urutan SPOK.
4. Jika ada kata Ocu yang tidak punya padanan jelas, pertahankan kata asli apa adanya.
5. Jangan menambahkan informasi yang tidak ada di input/referensi.
6. Output hanya SATU kalimat terjemahan, tanpa penjelasan."""

SUBJECT_WORDS_ID = {
    "saya", "aku", "kamu", "anda", "dia", "kami", "kita", "mereka",
    "orang", "anak", "istri", "suami", "ayah", "ibu"
}

NEGATION_WORDS_ID = {"tidak", "bukan", "belum", "jangan", "tak"}

class LLMTranslator:
    """Penerjemah berbasis LLM menggunakan API lokal Ollama."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Menginisialisasi penerjemah LLM.

        Parameter `api_key` dipertahankan untuk kompatibilitas mundur, tetapi tidak
        diperlukan untuk penggunaan Ollama lokal.
        """
        _ = api_key  # Mempertahankan tanda tangan kompatibel; tidak digunakan untuk Ollama lokal.
        base = OLLAMA_BASE_URL.rstrip("/") + "/"
        endpoint = OLLAMA_GENERATE_ENDPOINT.lstrip("/")
        self.generate_url = urljoin(base, endpoint)
        self.model = LLM_MODEL

    def _generate(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Memanggil endpoint generate Ollama dan mengembalikan respons teks."""
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        req = request.Request(
            self.generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"HTTP {exc.code} dari Ollama: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Tidak bisa terhubung ke Ollama: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Respons Ollama bukan JSON yang valid") from exc

        text = (data.get("response") or "").strip()
        if not text:
            raise ValueError("Ollama response kosong atau tidak valid")
        return text
        
    def translate(
        self, 
        text: str, 
        context: str,
        include_explanation: bool = False
    ) -> str:
        """Menerjemahkan teks Bahasa Ocu ke Bahasa Indonesia."""
        system_prompt = SYSTEM_PROMPT_OCU_TO_ID
        source_word_count = len([w for w in re.sub(r"[^\w\s]", "", text).split() if w])
        
        # Susun prompt pengguna.
        user_prompt = self._build_prompt(text, context, include_explanation)

        # Perakitan deterministik berbasis konteks retrieval ketika cakupan pemetaan kuat.
        if not include_explanation:
            deterministic = self._deterministic_ocu_to_id(text, context)
            if deterministic:
                return deterministic
        
        try:
            raw_translation = self._generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            cleaned_translation = self._sanitize_model_output(raw_translation)

            if not include_explanation:
                if self._is_unstable_long_output(cleaned_translation, source_word_count):
                    relaxed = self._deterministic_ocu_to_id(text, context, min_coverage=0.45)
                    if relaxed:
                        return relaxed
            
            # Pemrosesan pasca-terjemahan (post-processing): validasi struktur SPOK untuk Ocu->ID.
            # Lewati validasi SPOK untuk kata tunggal/frasa pendek (<=2 kata)
            # agar validator tidak menambah subjek yang tidak perlu.
            word_count = len(text.strip().split())
            if not include_explanation and word_count > 2:
                if self._needs_spok_validation(cleaned_translation):
                    return self._validate_spok(text, cleaned_translation)
                return self._format_output(cleaned_translation, False)
            
            return self._format_output(cleaned_translation, False)
            
        except Exception as e:
            if not include_explanation:
                relaxed = self._deterministic_ocu_to_id(text, context, min_coverage=0.35)
                if relaxed:
                    return relaxed
            return f"Error dalam penerjemahan: {str(e)}"

    def _deterministic_ocu_to_id(
        self,
        text: str,
        context: str,
        min_coverage: Optional[float] = None
    ) -> Optional[str]:
        """Menyusun terjemahan langsung dari pemetaan RAG saat cakupan tinggi."""
        mapping_map = self._extract_mapping_map(context)

        source_tokens = [tok for tok in re.sub(r"[^\w\s]", "", text.lower()).split() if tok]
        if not source_tokens:
            return None

        translated_tokens = []
        mapped_count = 0

        # Utamakan pemetaan n-gram terpanjang dari konteks untuk kalimat panjang.
        i = 0
        n_tokens = len(source_tokens)
        while i < n_tokens:
            matched = False
            for n in (3, 2, 1):
                if i + n > n_tokens:
                    continue
                key = " ".join(source_tokens[i:i + n])
                mapped = mapping_map.get(key)
                if mapped:
                    primary = mapped.split('/')[0].strip()
                    if primary:
                        translated_tokens.append(primary)
                        mapped_count += n
                        i += n
                        matched = True
                        break

            if matched:
                continue

            token = source_tokens[i]
            if token in {"dan", "atau", "yang", "di", "ke", "dengan"}:
                translated_tokens.append(token)
            else:
                # Pertahankan token yang tidak memiliki pemetaan retrieval.
                translated_tokens.append(token)
            i += 1

        if not translated_tokens:
            return None

        coverage = mapped_count / len(source_tokens)
        min_cov = min_coverage if min_coverage is not None else (0.60 if len(source_tokens) >= 5 else 0.75)
        if coverage < min_cov:
            return None

        candidate = " ".join(translated_tokens)
        candidate = self._dedupe_adjacent_tokens(candidate)
        candidate = self._light_spok_reorder(candidate)
        return self._format_output(candidate, False)

    def _extract_mapping_map(self, context: str) -> dict:
        """Ekstrak entri pemetaan (kata tunggal dan frasa pendek) dari konteks."""
        mappings = {}
        if not context:
            return mappings

        pattern = re.compile(r"^\s*-\s*'([^']+)'\s*=\s*'([^']+)'\s*$")
        for line in context.splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            source = match.group(1).strip().lower()
            
            # Hapus penjelasan kurung dari target untuk pencegahan (misal: "kau (wanita)" -> "kau")
            raw_target = match.group(2).strip()
            target = re.sub(r'\s*\([^)]*\)', '', raw_target).strip()
            
            # Simpan pemetaan hingga 3 token untuk perakitan deterministik berbasis frasa.
            src_tokens = [t for t in source.split() if t]
            tgt_tokens = [t for t in re.split(r"\W+", target.lower()) if t]

            if not (1 <= len(src_tokens) <= 3):
                continue

            # Pagar pengaman (guardrail): hindari pemetaan unit sumber pendek ke target kalimat panjang,
            # karena dapat menyisipkan klausa yang tidak relevan saat perakitan deterministik.
            max_tgt = len(src_tokens) + 3
            if len(src_tokens) == 1:
                max_tgt = 2
            if len(tgt_tokens) > max_tgt:
                continue

            mappings[source] = target

        return mappings

    def _light_spok_reorder(self, text: str) -> str:
        """Penyusunan ulang sederhana agar subjek dekat awal kalimat jika jelas salah posisi."""
        tokens = [tok for tok in text.split() if tok]
        if len(tokens) < 3:
            return text

        lower_tokens = [t.lower() for t in tokens]

        # Pindahkan subjek ke depan jika kalimat diawali negasi.
        if lower_tokens[0] in NEGATION_WORDS_ID:
            subj_idx = next((i for i, t in enumerate(lower_tokens[:5]) if t in SUBJECT_WORDS_ID), -1)
            if subj_idx > 0:
                subj = tokens.pop(subj_idx)
                tokens.insert(0, subj)
                return " ".join(tokens)

        # Jika subjek muncul terlambat namun token subjek Ocu ada di peta sumber,
        # biarkan kalimat apa adanya agar tidak terjadi koreksi berlebih.
        return " ".join(tokens)

    def _dedupe_adjacent_tokens(self, text: str) -> str:
        """Hapus duplikasi berurutan yang tidak sengaja (mis. 'anak anak')."""
        tokens = [t for t in text.split() if t]
        if not tokens:
            return text

        compact = [tokens[0]]
        for tok in tokens[1:]:
            if tok.lower() != compact[-1].lower():
                compact.append(tok)
        return " ".join(compact)

    def _is_unstable_long_output(self, text: str, source_word_count: int) -> bool:
        """Mendeteksi keluaran berpotensi halusinatif/terlalu panjang sebagai pagar pengaman kalimat panjang."""
        if not text:
            return True

        lowered = text.lower()
        if self._looks_like_instruction(lowered):
            return True

        out_tokens = [t for t in re.sub(r"[^\w\s]", "", lowered).split() if t]
        if not out_tokens:
            return True

        # Untuk kalimat sumber panjang, keluaran terlalu panjang sering menandakan penyimpangan.
        if source_word_count >= 5 and len(out_tokens) > max(18, int(source_word_count * 2.2)):
            return True

        return False
    
    def _validate_spok(self, original_ocu: str, translation: str) -> str:
        """
        Validasi dan perbaikan struktur SPOK pada terjemahan.
        SELALU kirim ke LLM untuk penyusunan ulang agar SPOK sesuai.
        Tanda baca asli dipertahankan: tidak menambah titik jika input tidak memilikinya.
        """
        if not translation or not translation.strip():
            return translation
        
        # Deteksi apakah masukan asli diakhiri tanda baca.
        original_has_punct = original_ocu.strip()[-1] in ('.', '!', '?') if original_ocu.strip() else False
        
        validation_prompt = (
            f"Kalimat berikut adalah terjemahan dari Bahasa Ocu ke Bahasa Indonesia.\n"
            f"Susun ulang agar mengikuti struktur SPOK yang benar.\n\n"
            f"Kalimat: \"{translation}\"\n\n"
            f"TUGAS: Susun ulang kalimat agar mengikuti struktur:\n"
            f"  Subjek + (Negasi) + Predikat + Objek + Keterangan\n\n"
            f"ATURAN KETAT:\n"
            f"1. Subjek (pelaku: saya, kamu, dia, orang, anak, istri, suami, dll) HARUS di awal kalimat\n"
            f"2. Kata negasi (tidak, bukan, belum, jangan) HARUS SETELAH Subjek\n"
            f"3. JANGAN mengubah makna kalimat\n"
            f"4. JANGAN menambah atau menghapus kata\n"
            f"5. Huruf pertama kalimat HARUS kapital\n"
            f"6. JANGAN menambahkan tanda baca (titik, tanda seru, tanda tanya) di akhir kalimat kecuali teks asli sudah memilikinya\n"
            f"7. Output HANYA satu kalimat hasil penyusunan ulang\n\n"
            f"Contoh:\n"
            f"  Input: 'tidak mau saya pergi dengan kamu'\n"
            f"  Output: Saya tidak mau pergi dengan kamu\n\n"
            f"  Input: 'belum selesai dia membuat tugas'\n"
            f"  Output: Dia belum selesai membuat tugas\n\n"
            f"  Input: 'orang tua membeli beras ke pasar pagi hari untuk kebutuhan rumah'\n"
            f"  Output: Orang tua membeli beras ke pasar pagi hari untuk kebutuhan rumah\n\n"
            f"  Input: 'istri membuat sambal di dapur rumah'\n"
            f"  Output: Istri membuat sambal di dapur rumah\n\n"
            f"Berikan HANYA kalimat yang sudah disusun ulang:"
        )
        
        try:
            validated = self._generate(
                user_prompt=validation_prompt,
                system_prompt=(
                    "Anda adalah validator tata bahasa Indonesia. "
                    "Tugas Anda HANYA menyusun ulang kalimat agar mengikuti "
                    "struktur SPOK (Subjek + Negasi + Predikat + Objek + Keterangan). "
                    "Subjek HARUS di awal kalimat. Huruf pertama HARUS kapital. "
                    "JANGAN menambahkan tanda baca di akhir kalimat kecuali teks asli "
                    "sudah memilikinya. JANGAN mengubah makna. Output HANYA satu kalimat."
                ),
                temperature=0.05,
                max_tokens=256
            )
            # Sanitasi untuk mencegah kebocoran instruksi dari model kecil.
            validated = self._sanitize_model_output(validated)
            if self._looks_like_instruction(validated):
                return self._format_output(translation, original_has_punct)
            # Pastikan format benar (menghormati tanda baca asli)
            validated = self._format_output(validated, original_has_punct)
            return validated
        except Exception:
            # Jika validasi gagal, minimal rapikan keluaran asli
            return self._format_output(translation, original_has_punct)

    def _looks_like_instruction(self, text: str) -> bool:
        """Mendeteksi keluaran model yang berupa instruksi, bukan terjemahan."""
        if not text:
            return True
        lowered = text.lower()
        bad_markers = (
            "susun ulang", "aturan", "tugas:", "input:", "output:", "struktur spok", "contoh:"
        )
        return any(marker in lowered for marker in bad_markers)
    
    def _format_output(self, text: str, add_punct: bool = False) -> str:
        """Pastikan kapitalisasi benar. Tambah tanda baca hanya jika add_punct bernilai True."""
        if not text or not text.strip():
            return text
        
        # Lapisan penjagaan akhir: bersihkan kurung penjabaran sebelum dikembalikan.
        text = re.sub(r'\s*\([^)]*\)', '', text).strip()
            
        text = text.strip()
        # Kapitalisasi huruf pertama
        text = text[0].upper() + text[1:]
        # Tambahkan titik hanya jika masukan asli memiliki tanda baca akhir.
        if add_punct and not text.endswith(('.', '!', '?')):
            text += '.'
        # Hapus titik akhir jika masukan asli tidak memiliki tanda baca akhir.
        if not add_punct and text.endswith('.'):
            text = text.rstrip('.')
        return text

    def _sanitize_model_output(self, text: str) -> str:
        """Menormalisasi keluaran model menjadi satu baris terjemahan yang bersih."""
        if not text:
            return text

        cleaned = text.strip().strip('"').strip("'").strip()
        cleaned = cleaned.replace("\r", "\n")

        # Hapus penjelasan tambahan yang ada di dalam tanda kurung, seperti "(wanita)"
        cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned).strip()

        # Hapus prefiks meta umum yang sering dihasilkan model kecil.
        cleaned = re.sub(r"^(terjemahan\s*:|jawaban\s*:|output\s*:)", "", cleaned, flags=re.IGNORECASE).strip()

        lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
        if not lines:
            return ""

        blocked_prefixes = (
            "referensi", "arti:", "analisis", "penjelasan", "input:", "output:", "catatan"
        )
        candidates = [ln for ln in lines if not ln.lower().startswith(blocked_prefixes)]
        chosen = candidates[0] if candidates else lines[0]

        # Jika model menghasilkan banyak kalimat, ambil kalimat pertama.
        sentence_split = re.split(r"(?<=[.!?])\s+", chosen)
        chosen = sentence_split[0].strip()

        # Hapus penanda daftar/bullet.
        chosen = re.sub(r"^[-*\d\.)\s]+", "", chosen).strip()
        return chosen

    def _needs_spok_validation(self, translation: str) -> bool:
        """Gerbang heuristik untuk menghindari panggilan LLM kedua yang tidak perlu."""
        if not translation:
            return False

        tokens = [t for t in re.sub(r"[^\w\s]", "", translation.lower()).split() if t]
        if len(tokens) < 3:
            return False

        first = tokens[0]
        first_three: Set[str] = set(tokens[:3])

        # Jika kalimat diawali negasi dan subjek muncul belakangan, perlu penyusunan ulang.
        if first in NEGATION_WORDS_ID and first_three.isdisjoint(SUBJECT_WORDS_ID):
            return True

        # Jika subjek dikenal muncul di 2 token pertama, kemungkinan sudah baik.
        if set(tokens[:2]) & SUBJECT_WORDS_ID:
            return False

        # Jika tidak ada subjek pada tiga token pertama, minta validator.
        return first_three.isdisjoint(SUBJECT_WORDS_ID)
    
    def _build_prompt(
        self, 
        text: str, 
        context: str, 
        include_explanation: bool
    ) -> str:
        """Menyusun user prompt untuk penerjemahan."""
        source_lang = "Bahasa Ocu"
        target_lang = "Bahasa Indonesia"
            
        prompt_parts = [
            "Gunakan referensi berikut sebagai satu-satunya acuan utama.",
            "=== REFERENSI RAG MULAI ===",
            context,
            "=== REFERENSI RAG SELESAI ===",
            "",
            f"Teks {source_lang}:",
            f'"{text}"',
            ""
        ]
        
        word_count = len(text.strip().split())

        if include_explanation:
            prompt_parts.extend([
                f"Berikan terjemahan dalam {target_lang} beserta penjelasan arti kata per kata."
            ])
        else:
            if word_count <= 2:
                # Untuk kata tunggal/frasa pendek: keluarkan terjemahan langsung saja,
                # tanpa subjek tambahan, tanpa kalimat penuh, tanpa penyusunan ulang SPOK.
                prompt_parts.extend([
                    f"Terjemahkan ke {target_lang}.",
                    "",
                    "INSTRUKSI KRITIS:",
                    "1. Berikan HANYA terjemahan langsung â€” kata atau frasa saja.",
                    "2. JANGAN menambahkan subjek atau membuat kalimat lengkap.",
                    "3. JANGAN menambahkan penjelasan, definisi, atau kata tambahan.",
                    "4. Output hanya satu baris.",
                ])
            else:
                prompt_parts.extend([
                    f"Terjemahkan ke {target_lang}.",
                    "",
                    "INSTRUKSI KRITIS â€” WAJIB DIIKUTI:",
                    "1. Berikan HANYA satu kalimat terjemahan dalam Bahasa Indonesia baku",
                    "2. GUNAKAN referensi untuk memahami arti setiap kata Ocu",
                    "3. JANGAN SALIN urutan kata dari Bahasa Ocu atau dari referensi kalimat",
                    "4. Jika kalimat referensi bertanda [MIRIP, BUKAN INPUT ANDA], JANGAN SALIN terjemahannya",
                    "5. Kata yang tidak ada di referensi:",
                    "   - Kata Indonesia umum (tadi, pagi, sudah, dll) â†’ pertahankan apa adanya",
                    "   - Kata Ocu yang tidak dikenal â†’ pertahankan kata aslinya",
                    "6. JANGAN menambahkan kata yang tidak ada hubungannya dengan input",
                    "7. Output hanya SATU kalimat tanpa label seperti 'Terjemahan:' atau 'Arti:'",
                    "",
                    "PROSEDUR PENYUSUNAN KALIMAT (WAJIB):",
                    "Setelah menerjemahkan setiap kata, SUSUN ULANG dengan aturan:",
                    "  a. Temukan kata PELAKU (Subjek) â†’ letakkan di AWAL kalimat",
                    "  b. Jika ada NEGASI (indak=tidak, bukan, dll) â†’ letakkan SETELAH Subjek",
                    "  c. Letakkan PREDIKAT (kata kerja) setelah negasi/subjek",
                    "  d. Letakkan OBJEK setelah predikat",
                    "  e. Letakkan KETERANGAN di akhir kalimat",
                    "  Hasil: Subjek + (Negasi) + Predikat + Objek + Keterangan"
                ])

        return "\n".join(prompt_parts)
    
    def translate_simple(self, text: str) -> str:
        """
        Terjemahan sederhana tanpa konteks RAG (tidak direkomendasikan)
        
        Parameter:
            text: Teks masukan dalam Bahasa Ocu.
            
        Mengembalikan:
            Teks hasil terjemahan.
        """
        prompt = (
            f"Terjemahkan teks Bahasa Ocu berikut ke Bahasa Indonesia:\n"
            f'"{text}"\n\n'
            f"Berikan hanya terjemahannya."
        )
        
        try:
            return self._generate(
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT_OCU_TO_ID,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Uji LLM (membutuhkan server Ollama lokal).
    print("Testing LLM translator...")
    print("Note: This requires Ollama running at configured endpoint")
    
    try:
        translator = LLMTranslator()
        
        test_text = "deyen poi ka kampuong"
        test_context = """Referensi terjemahan yang relevan:
  - [Kalimat] 'deyen poi ka kampuong' = 'saya pergi ke kampung' (kalimat deklaratif)
  - [Kosakata] 'deyen' = 'saya' (pronomina)
  
Arti kata per kata:
  - 'deyen' = 'saya'
  - 'poi' = 'pergi'
  - 'ka' = 'ke'
  - 'kampuong' = 'kampung'"""
        
        result = translator.translate(test_text, test_context)
        print(f"\nInput: {test_text}")
        print(f"Output: {result}")
        
    except ValueError as e:
        print(f"Skipping test: {e}")
