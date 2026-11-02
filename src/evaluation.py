"""
Komponen pendukung penelitian - Evaluasi Kualitas Terjemahan dan Pengambilan Informasi (Retrieval).
Menyediakan metrik kuantitatif untuk menilai performa terjemahan
serta kualitas pengambilan konteks.
"""
import re
import json
import urllib.request as request
import urllib.error as error
from urllib.parse import urljoin
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.config import OLLAMA_BASE_URL, OLLAMA_GENERATE_ENDPOINT, LLM_MODEL


class TranslationEvaluator:
    """
    Evaluator untuk kualitas terjemahan.
    Mencakup beberapa metrik: BLEU, akurasi, kemiripan semantik, dan lainnya.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        """Menginisialisasi evaluator dengan model penyematan vektor."""
        if embedding_model is not None:
            print("Using supplied embedding model for evaluation")
            self.embedding_model = embedding_model
        else:
            print(f"Loading embedding model for evaluation: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def normalize_text(self, text: str) -> str:
        """Menormalisasi teks untuk perbandingan."""
        # Ubah menjadi huruf kecil.
        text = text.lower().strip()
        # Hapus spasi berlebih.
        text = re.sub(r'\s+', ' ', text)
        # Hapus tanda baca.
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _token_overlap(self, prediction: str, reference: str) -> float:
        """Menghitung tumpang tindih F1 tingkat token antara prediksi dan referensi."""
        pred_tokens = self.tokenize(prediction)
        ref_tokens = self.tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        matched = sum((pred_counter & ref_counter).values())

        precision = matched / len(pred_tokens) if pred_tokens else 0.0
        recall = matched / len(ref_tokens) if ref_tokens else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _semantic_similarity(self, prediction: str, reference: str) -> float:
        """Menghitung kemiripan kosinus menggunakan penyematan kalimat."""
        pred_norm = prediction.strip()
        ref_norm = reference.strip()
        if not pred_norm or not ref_norm:
            return 0.0

        try:
            embeddings = self.embedding_model.encode([pred_norm, ref_norm], convert_to_numpy=True, normalize_embeddings=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity.item())
        except Exception:
            return 0.0
    
    def tokenize(self, text: str) -> List[str]:
        """Melakukan tokenisasi kata sederhana."""
        return self.normalize_text(text).split()
    
    def calculate_bleu(
        self, 
        prediction: str, 
        reference: str, 
        n_gram: int = 4
    ) -> Dict[str, float]:
        """
        Menghitung skor BLEU (versi sederhana).
        Mengembalikan BLEU-1, BLEU-2, BLEU-3, dan BLEU-4.
        """
        pred_tokens = self.tokenize(prediction)
        ref_tokens = self.tokenize(reference)
        
        if not pred_tokens or not ref_tokens:
            return {f"bleu_{i}": 0.0 for i in range(1, n_gram + 1)}
        
        bleu_scores = {}
        
        for n in range(1, n_gram + 1):
            # Ambil n-gram.
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not ref_ngrams:
                bleu_scores[f"bleu_{n}"] = 0.0
                continue
            
            # Hitung kecocokan.
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            # Hitung jumlah terpotong.
            clipped_counts = sum((pred_counter & ref_counter).values())
            total_pred = len(pred_ngrams)
            
            # Presisi.
            precision = clipped_counts / total_pred if total_pred > 0 else 0
            bleu_scores[f"bleu_{n}"] = precision
        
        # Hitung rata-rata BLEU.
        bleu_scores["bleu_avg"] = np.mean(list(bleu_scores.values()))
        
        return bleu_scores
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Mengambil n-gram dari daftar token."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def evaluate_single(
        self, 
        prediction: str, 
        reference: str,
        input_text: str = None
    ) -> Dict[str, float]:
        """
        Mengevaluasi satu terjemahan.
        
        Parameter:
            prediction: Terjemahan model.
            reference: Terjemahan acuan.
            input_text: Input asli (opsional).
            
        Mengembalikan:
            Kamus metrik.
        """
        metrics = {}

        normalized_prediction = self.normalize_text(prediction)
        normalized_reference = self.normalize_text(reference)

        metrics["exact_match"] = 1.0 if normalized_prediction == normalized_reference and normalized_prediction else 0.0
        metrics["word_accuracy"] = self._token_overlap(prediction, reference)
        metrics["semantic_similarity"] = self._semantic_similarity(prediction, reference)
        
        # Skor BLEU.
        bleu_scores = self.calculate_bleu(prediction, reference)
        metrics.update(bleu_scores)
        
        # Skor LLM sebagai juri.
        metrics["llm_judge_score"] = self.llm_judge(prediction, reference)
        
        return metrics

    def llm_judge(self, prediction: str, reference: str) -> float:
        """Menggunakan LLM lokal (Ollama) untuk menilai terjemahan dari 1 sampai 5."""
        base = OLLAMA_BASE_URL.rstrip("/") + "/"
        endpoint = OLLAMA_GENERATE_ENDPOINT.lstrip("/")
        generate_url = urljoin(base, endpoint)
        
        system_prompt = "Anda adalah juri profesional untuk menilai kualitas terjemahan."
        user_prompt = f"""Bandinkan terjemahan prediksi ini dengan referensinya. 
Beri skor angka bulat (1 sampai 5), di mana 1 artinya salah sepenuhnya, dan 5 artinya sempurna (sama secara makna).
Referensi: "{reference}"
Prediksi: "{prediction}"
Jawaban harus HANYA SATU ANGKA (misal: 4).
Skor:"""

        payload = {
            "model": LLM_MODEL,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 5
            }
        }

        req = request.Request(
            generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                response_text = data.get("response", "").strip()
                # Temukan digit pertama.
                match = re.search(r'\d+', response_text)
                if match:
                    score = float(match.group())
                    return max(1.0, min(5.0, score))
                else:
                    return 0.0
        except Exception as exc:
            pass
        return 0.0
    
    def evaluate_batch(
        self, 
        predictions: List[str], 
        references: List[str],
        input_texts: List[str] = None
    ) -> Dict[str, float]:
        """
        Mengevaluasi beberapa terjemahan dan mengembalikan metrik agregat.
        
        Parameter:
            predictions: Daftar terjemahan model.
            references: Daftar terjemahan acuan.
            input_texts: Daftar input asli (opsional).
            
        Mengembalikan:
            Kamus metrik agregat.
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
        
        all_metrics = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            input_text = input_texts[i] if input_texts else None
            metrics = self.evaluate_single(pred, ref, input_text)
            all_metrics.append(metrics)
        
        # Agregasikan metrik.
        aggregated = {}
        
        # Ambil seluruh nama metrik.
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            aggregated[f"{metric_name}_mean"] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_min"] = np.min(values)
            aggregated[f"{metric_name}_max"] = np.max(values)
        
        # Tambahkan jumlah sampel.
        aggregated["num_samples"] = len(predictions)
        
        return aggregated, all_metrics


class RetrievalEvaluator:
    """Evaluator untuk kualitas pengambilan informasi RAG."""
    
    def __init__(self):
        """Menginisialisasi evaluator pengambilan informasi."""
        pass
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_items: List[Dict],
        ground_truth_items: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Mengevaluasi kualitas pengambilan informasi.
        
        Parameter:
            query: Teks kueri.
            retrieved_items: Daftar entri yang diambil beserta metadata.
            ground_truth_items: Daftar entri relevan yang diharapkan (opsional).
            top_k: Jumlah hasil teratas yang dipertimbangkan.
            
        Mengembalikan:
            Kamus metrik pengambilan informasi.
        """
        metrics = {}
        
        # Jika acuan tersedia, hitung presisi/recall.
        if ground_truth_items:
            # Sertakan teks Ocu dan Indonesia dari hasil pengambilan agar pencocokan fleksibel.
            retrieved_texts = set()
            for item in retrieved_items[:top_k]:
                ocu = item.get("ocu_text", "").lower().strip()
                indo = item.get("indo_text", "").lower().strip()
                if ocu: retrieved_texts.add(ocu)
                if indo: retrieved_texts.add(indo)
                
            ground_truth_texts = set(gt.lower().strip() for gt in ground_truth_items)
            
            # Positif benar (jika teks yang diambil cocok dengan teks acuan mana pun).
            tp = len(retrieved_texts & ground_truth_texts)
            
            # Hitung penyebut pengambilan berdasarkan jumlah entri unik yang diambil.
            num_retrieved_items = len(retrieved_items[:top_k])
            
            # Catat peringkat untuk menghitung MRR.
            first_relevant_rank = 0
            for rank, item in enumerate(retrieved_items[:top_k], start=1):
                item_texts = {item.get("ocu_text", "").lower().strip(), 
                              item.get("indo_text", "").lower().strip()}
                if item_texts & ground_truth_texts:
                    first_relevant_rank = rank
                    break
            
            # Peringkat Resiprokal Rata-rata (Mean Reciprocal Rank/MRR).
            metrics["mrr"] = (1.0 / first_relevant_rank) if first_relevant_rank > 0 else 0.0
            
            # Presisi@K (Precision@K): TP / entri yang diambil (dibatasi hingga 1.0).
            metrics[f"precision_at_{top_k}"] = min(1.0, tp / num_retrieved_items) if num_retrieved_items else 0
            
            # Perolehan kembali@K (Recall@K): TP / acuan.
            metrics[f"recall_at_{top_k}"] = min(1.0, tp / len(ground_truth_texts)) if ground_truth_texts else 0
            
            # Skor F1 berdasarkan @K.
            p_k = metrics[f"precision_at_{top_k}"]
            r_k = metrics[f"recall_at_{top_k}"]
            if p_k + r_k > 0:
                metrics["f1_score"] = 2 * (p_k * r_k) / (p_k + r_k)
            else:
                metrics["f1_score"] = 0
                
            # Pertahankan nama umum untuk kompatibilitas mundur.
            metrics["precision"] = metrics[f"precision_at_{top_k}"]
            metrics["recall"] = metrics[f"recall_at_{top_k}"]
        
        return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Menampilkan metrik evaluasi dengan format yang mudah dibaca."""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{metric_name:.<40} {value:.4f}")
        else:
            print(f"{metric_name:.<40} {value}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Uji evaluator.
    print("Testing Translation Evaluator...")
    
    evaluator = TranslationEvaluator()
    
    # Uji evaluasi tunggal.
    prediction = "saya pergi ke kampung"
    reference = "saya pergi ke kampung"
    
    metrics = evaluator.evaluate_single(prediction, reference)
    print_metrics(metrics, "Single Translation Evaluation")
    
    # Uji dengan prediksi yang berbeda.
    prediction2 = "aku pergi kampung"
    metrics2 = evaluator.evaluate_single(prediction2, reference)
    print_metrics(metrics2, "Different Translation Evaluation")
    
    # Uji evaluasi kumpulan data.
    predictions = [
        "saya pergi ke kampung",
        "kamu makan nasi",
        "orang tua membeli beras"
    ]
    references = [
        "saya pergi ke kampung",
        "kamu makan nasi",
        "orang tua membeli beras di pasar"
    ]
    
    agg_metrics, individual = evaluator.evaluate_batch(predictions, references)
    print_metrics(agg_metrics, "Batch Evaluation - Aggregated")
