"""
Tahap 10 - Evaluasi End-to-End OCU ke Indonesia.
Menjalankan pengujian batch pada test set dan menyimpan ringkasan metrik.
"""
import os
import sys
import csv
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translator import create_translator
from src.evaluation import TranslationEvaluator, RetrievalEvaluator, print_metrics


ACTIVE_METRICS = {
    'BLEU-4': 'bleu_4',
    'Semantic Similarity': 'semantic_similarity',
    'LLM Judge': 'llm_judge_score'
}


def load_test_set(test_file: str = "dataset/test_set.csv") -> List[Dict]:
    """
    Load test dataset
    
    Returns:
        List of test cases with input and reference translation
    """
    test_cases = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            test_cases.append({
                'id': row['ID'],
                'input': row['Bahasa Ocu'],
                'reference': row['Terjemahan Indonesia'],
                'type': row['Tipe'],
                'keterangan': row['Keterangan']
            })
    
    return test_cases


def run_translation_evaluation(
    translator,
    test_cases: List[Dict],
    output_dir: str = "evaluation_results",
    run_metadata: Optional[Dict] = None,
) -> Dict:
    """
    Run translation evaluation on test set
    
    Args:
        translator: OcuTranslator instance
        test_cases: List of test cases
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    print("\n" + "="*70)
    print("EVALUASI SISTEM PENERJEMAHAN BAHASA OCU".center(70))
    print("="*70)
    print(f"\nJumlah test cases: {len(test_cases)}")
    print(f"Tanggal evaluasi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*70)
    
    # Inisialisasi evaluator
    print("\nMemuat evaluator...")
    translator_retriever = getattr(translator, 'retriever', None)
    translator_store = getattr(translator_retriever, 'vector_store', None)
    store_embedding = getattr(translator_store, 'embedding_model', None)
    shared_embedding_model = getattr(store_embedding, 'model', None)
    if shared_embedding_model is None:
        eval_engine = TranslationEvaluator()
    else:
        eval_engine = TranslationEvaluator(embedding_model=shared_embedding_model)
    retrieval_eval = RetrievalEvaluator()
    
    # Simpan hasil
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_test_cases': len(test_cases),
        'run_metadata': copy.deepcopy(run_metadata or {}),
        'test_cases': [],
        'overall_metrics': {},
        'metrics_by_type': {}
    }
    
    predictions = []
    references = []
    inputs = []
    
    # Jalankan proses terjemahan
    print("\n" + "="*70)
    print("MENERJEMAHKAN TEST CASES".center(70))
    print("="*70 + "\n")
    
    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case['id']
        input_text = test_case['input']
        reference = test_case['reference']
        test_type = test_case['type']
        
        print(f"\n[{i}/{len(test_cases)}] {test_id} ({test_type})")
        print(f"Input    : {input_text}")
        print(f"Reference: {reference}")
        
        # Terjemahkan
        try:
            translation_result = translator.translate(input_text)
            if not getattr(translation_result, "success", True):
                error_message = getattr(translation_result, "error_message", None) or "translation failed"
                print(f"  ✗ ERROR: {error_message}")
                results['test_cases'].append({
                    'id': test_id,
                    'type': test_type,
                    'input': input_text,
                    'reference': reference,
                    'prediction': None,
                    'error': error_message
                })
                continue
            prediction = translation_result.translated_text
            retrieval_context = translation_result.retrieval_results
            
            print(f"Prediksi : {prediction}")
            
            # Evaluasi terjemahan tunggal
            metrics = eval_engine.evaluate_single(prediction, reference, input_text)
            
            # Ubah hasil retrieval ke format dict untuk evaluasi
            retrieval_items = []
            for ret_result in retrieval_context:
                retrieval_items.append({
                    'ocu_text': ret_result.entry.bahasa_ocu,
                    'indo_text': ret_result.entry.bahasa_indonesia,
                    'type': ret_result.entry.entry_type,
                    'similarity': ret_result.score
                })
            
            # Evaluasi kualitas retrieval memakai input dan keluaran harapan sebagai ground truth
            retrieval_metrics = retrieval_eval.evaluate_retrieval(
                query=input_text,
                retrieved_items=retrieval_items,
                ground_truth_items=[input_text, reference],
                top_k=5
            )
            if 'avg_similarity' not in retrieval_metrics:
                similarities = [item['similarity'] for item in retrieval_items]
                retrieval_metrics['avg_similarity'] = (
                    sum(similarities) / len(similarities) if similarities else 0.0
                )
            
            # Tampilkan metrik ringkas
            print(f"  ✓ Exact Match: {metrics.get('exact_match', 0.0):.3f}")
            print(f"  ✓ Word Accuracy: {metrics.get('word_accuracy', 0.0):.3f}")
            print(f"  ✓ BLEU-4: {metrics.get('bleu_4', 0.0):.3f}")
            print(f"  ✓ Semantic Sim: {metrics.get('semantic_similarity', 0.0):.3f}")
            print(f"  ✓ LLM Judge: {metrics['llm_judge_score']:.1f}/5")
            print(f"  ✓ Retrieval Sim: {retrieval_metrics['avg_similarity']:.3f}")
            print(f"  ✓ Retrieval Precision: {retrieval_metrics.get('precision', 0.0):.3f}")
            
            # Simpan hasil per kasus uji
            test_result = {
                'id': test_id,
                'type': test_type,
                'input': input_text,
                'reference': reference,
                'prediction': prediction,
                'translation_metrics': metrics,
                'retrieval_metrics': retrieval_metrics,
                'num_retrieved': len(retrieval_context)
            }
            
            results['test_cases'].append(test_result)
            predictions.append(prediction)
            references.append(reference)
            inputs.append(input_text)
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results['test_cases'].append({
                'id': test_id,
                'type': test_type,
                'input': input_text,
                'reference': reference,
                'prediction': None,
                'error': str(e)
            })
    
    # Hitung metrik keseluruhan
    print("\n\n" + "="*70)
    print("MENGHITUNG METRIK KESELURUHAN".center(70))
    print("="*70)
    
    if predictions:
        overall_metrics, individual_metrics = eval_engine.evaluate_batch(
            predictions, references, inputs
        )
        results['overall_metrics'] = overall_metrics
        
        # Tampilkan metrik keseluruhan
        print("\n" + "="*60)
        print(f"{'METRIK KESELURUHAN (AKTIF)':^60}")
        print("="*60)
        print(f"exact_match_mean{'':.<35} {overall_metrics.get('exact_match_mean', 0.0):.4f}")
        print(f"word_accuracy_mean{'':.<33} {overall_metrics.get('word_accuracy_mean', 0.0):.4f}")
        print(f"bleu_4_mean{'':.<38} {overall_metrics.get('bleu_4_mean', 0.0):.4f}")
        print(f"semantic_similarity_mean{'':.<27} {overall_metrics.get('semantic_similarity_mean', 0.0):.4f}")
        print(f"llm_judge_score_mean{'':.<31} {overall_metrics.get('llm_judge_score_mean', 0.0):.4f}")
        print("="*60 + "\n")
    else:
        print("\nTidak ada prediksi yang berhasil untuk dievaluasi.")
        results['overall_metrics'] = {}
        individual_metrics = []

    overall_metrics = results['overall_metrics']
    
    # Hitung metrik per tipe
    print("\n" + "="*70)
    print("METRIK PER KATEGORI".center(70))
    print("="*70 + "\n")
    
    types = set(tc['type'] for tc in test_cases)
    for test_type in sorted(types):
        type_predictions = []
        type_references = []
        
        for tc in results['test_cases']:
            if tc['type'] == test_type and tc.get('prediction'):
                type_predictions.append(tc['prediction'])
                type_references.append(tc['reference'])
        
        if type_predictions:
            type_metrics, _ = eval_engine.evaluate_batch(
                type_predictions, type_references
            )
            results['metrics_by_type'][test_type] = type_metrics
            
            print(f"\n{test_type.upper()}")
            print("-" * 70)
            print(f"  Samples: {len(type_predictions)}")
            print(f"  Exact Match: {type_metrics.get('exact_match_mean', 0.0):.3f}")
            print(f"  Word Accuracy: {type_metrics.get('word_accuracy_mean', 0.0):.3f}")
            print(f"  BLEU-4: {type_metrics['bleu_4_mean']:.3f}")
            print(f"  Semantic Similarity: {type_metrics['semantic_similarity_mean']:.3f}")
            print(f"  LLM Judge Score: {type_metrics.get('llm_judge_score_mean', 0.0):.3f}/5")
    
    # Simpan hasil
    print("\n" + "="*70)
    print("MENYIMPAN HASIL".center(70))
    print("="*70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simpan hasil JSON terperinci
    json_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    latest_json_file = os.path.join(output_dir, "latest_evaluation.json")
    with open(latest_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Hasil detail disimpan: {json_file}")
    print(f"✓ Hasil detail terbaru disimpan: {latest_json_file}")
    
    # Simpan laporan ringkasan
    report_file = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LAPORAN EVALUASI SISTEM PENERJEMAHAN BAHASA OCU\n")
        f.write("="*70 + "\n\n")
        f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Jumlah Test Cases: {len(test_cases)}\n\n")

        if results['run_metadata']:
            f.write("METADATA EKSPERIMEN\n")
            f.write("-"*70 + "\n")
            for key in sorted(results['run_metadata']):
                f.write(f"{key}: {results['run_metadata'][key]}\n")
            f.write("\n")
        
        f.write("METRIK KESELURUHAN\n")
        f.write("-"*70 + "\n")
        f.write(f"{'exact_match_mean':.<50} {overall_metrics.get('exact_match_mean', 0.0):.4f}\n")
        f.write(f"{'word_accuracy_mean':.<50} {overall_metrics.get('word_accuracy_mean', 0.0):.4f}\n")
        f.write(f"{'bleu_4_mean':.<50} {overall_metrics.get('bleu_4_mean', 0.0):.4f}\n")
        f.write(f"{'semantic_similarity_mean':.<50} {overall_metrics.get('semantic_similarity_mean', 0.0):.4f}\n")
        f.write(f"{'llm_judge_score_mean':.<50} {overall_metrics.get('llm_judge_score_mean', 0.0):.4f}\n")
        
        f.write("\n\nMETRIK PER KATEGORI\n")
        f.write("-"*70 + "\n")
        for test_type, type_metrics in sorted(results['metrics_by_type'].items()):
            f.write(f"\n{test_type.upper()}\n")
            f.write(f"  Exact Match: {type_metrics.get('exact_match_mean', 0.0):.3f}\n")
            f.write(f"  Word Accuracy: {type_metrics.get('word_accuracy_mean', 0.0):.3f}\n")
            f.write(f"  BLEU-4: {type_metrics['bleu_4_mean']:.3f}\n")
            f.write(f"  Semantic Similarity: {type_metrics['semantic_similarity_mean']:.3f}\n")
            if 'llm_judge_score_mean' in type_metrics:
                f.write(f"  LLM Judge Score: {type_metrics['llm_judge_score_mean']:.3f}/5\n")
        
        f.write("\n\nDETAIL PER TEST CASE\n")
        f.write("="*70 + "\n")
        for tc in results['test_cases']:
            if tc.get('prediction'):
                f.write(f"\n[{tc['id']}] {tc['type']}\n")
                f.write(f"Input     : {tc['input']}\n")
                f.write(f"Reference : {tc['reference']}\n")
                f.write(f"Prediction: {tc['prediction']}\n")
                f.write(f"Exact Match: {tc['translation_metrics'].get('exact_match', 0.0):.3f} | ")
                f.write(f"Word Accuracy: {tc['translation_metrics'].get('word_accuracy', 0.0):.3f} | ")
                f.write(f"BLEU-4: {tc['translation_metrics'].get('bleu_4', 0.0):.3f} | ")
                f.write(f"Semantic: {tc['translation_metrics'].get('semantic_similarity', 0.0):.3f} | ")
                f.write(f"LLM Judge: {tc['translation_metrics'].get('llm_judge_score', 0):.1f}/5\n")
                if 'precision' in tc['retrieval_metrics']:
                    f.write(f"Retrieval Precision: {tc['retrieval_metrics']['precision']:.3f}\n")
    
    latest_report_file = os.path.join(output_dir, "latest_report.txt")
    with open(latest_report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LAPORAN EVALUASI SISTEM PENERJEMAHAN BAHASA OCU\n")
        f.write("="*70 + "\n\n")
        f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Jumlah Test Cases: {len(test_cases)}\n\n")

        if results['run_metadata']:
            f.write("METADATA EKSPERIMEN\n")
            f.write("-"*70 + "\n")
            for key in sorted(results['run_metadata']):
                f.write(f"{key}: {results['run_metadata'][key]}\n")
            f.write("\n")
        
        f.write("METRIK KESELURUHAN\n")
        f.write("-"*70 + "\n")
        f.write(f"{'exact_match_mean':.<50} {overall_metrics.get('exact_match_mean', 0.0):.4f}\n")
        f.write(f"{'word_accuracy_mean':.<50} {overall_metrics.get('word_accuracy_mean', 0.0):.4f}\n")
        f.write(f"{'bleu_4_mean':.<50} {overall_metrics.get('bleu_4_mean', 0.0):.4f}\n")
        f.write(f"{'semantic_similarity_mean':.<50} {overall_metrics.get('semantic_similarity_mean', 0.0):.4f}\n")
        f.write(f"{'llm_judge_score_mean':.<50} {overall_metrics.get('llm_judge_score_mean', 0.0):.4f}\n")
        
        f.write("\n\nMETRIK PER KATEGORI\n")
        f.write("-"*70 + "\n")
        for test_type, type_metrics in sorted(results['metrics_by_type'].items()):
            f.write(f"\n{test_type.upper()}\n")
            f.write(f"  Exact Match: {type_metrics.get('exact_match_mean', 0.0):.3f}\n")
            f.write(f"  Word Accuracy: {type_metrics.get('word_accuracy_mean', 0.0):.3f}\n")
            f.write(f"  BLEU-4: {type_metrics['bleu_4_mean']:.3f}\n")
            f.write(f"  Semantic Similarity: {type_metrics['semantic_similarity_mean']:.3f}\n")
            if 'llm_judge_score_mean' in type_metrics:
                f.write(f"  LLM Judge Score: {type_metrics['llm_judge_score_mean']:.3f}/5\n")
        
        f.write("\n\nDETAIL PER TEST CASE\n")
        f.write("="*70 + "\n")
        for tc in results['test_cases']:
            if tc.get('prediction'):
                f.write(f"\n[{tc['id']}] {tc['type']}\n")
                f.write(f"Input     : {tc['input']}\n")
                f.write(f"Reference : {tc['reference']}\n")
                f.write(f"Prediction: {tc['prediction']}\n")
                f.write(f"Exact Match: {tc['translation_metrics'].get('exact_match', 0.0):.3f} | ")
                f.write(f"Word Accuracy: {tc['translation_metrics'].get('word_accuracy', 0.0):.3f} | ")
                f.write(f"BLEU-4: {tc['translation_metrics'].get('bleu_4', 0.0):.3f} | ")
                f.write(f"Semantic: {tc['translation_metrics'].get('semantic_similarity', 0.0):.3f} | ")
                f.write(f"LLM Judge: {tc['translation_metrics'].get('llm_judge_score', 0):.1f}/5\n")
                if 'precision' in tc['retrieval_metrics']:
                    f.write(f"Retrieval Precision: {tc['retrieval_metrics']['precision']:.3f}\n")
    print(f"✓ Laporan ringkasan disimpan: {report_file}")
    print(f"✓ Laporan ringkasan terbaru disimpan: {latest_report_file}")
    
    # Simpan ringkasan CSV
    csv_file = os.path.join(output_dir, f"summary_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['ID', 'Type', 'Input', 'Reference', 'Prediction',
                     'Exact_Match', 'Word_Accuracy', 'BLEU_4', 'Semantic_Similarity',
                     'LLM_Judge', 'Retrieval_Precision']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tc in results['test_cases']:
            if tc.get('prediction'):
                writer.writerow({
                    'ID': tc['id'],
                    'Type': tc['type'],
                    'Input': tc['input'],
                    'Reference': tc['reference'],
                    'Prediction': tc['prediction'],
                    'Exact_Match': tc['translation_metrics'].get('exact_match', 0.0),
                    'Word_Accuracy': tc['translation_metrics'].get('word_accuracy', 0.0),
                    'BLEU_4': tc['translation_metrics'].get('bleu_4', 0.0),
                    'Semantic_Similarity': tc['translation_metrics'].get('semantic_similarity', 0.0),
                    'LLM_Judge': tc['translation_metrics'].get('llm_judge_score', 0.0),
                    'Retrieval_Precision': tc['retrieval_metrics'].get('precision', 0.0)
                })
    
    latest_csv_file = os.path.join(output_dir, "latest_summary.csv")
    with open(latest_csv_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['ID', 'Type', 'Input', 'Reference', 'Prediction', 
                     'Exact_Match', 'Word_Accuracy', 'BLEU_4', 'Semantic_Similarity',
                     'LLM_Judge', 'Retrieval_Precision']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for tc in results['test_cases']:
            if tc.get('prediction'):
                writer.writerow({
                    'ID': tc['id'],
                    'Type': tc['type'],
                    'Input': tc['input'],
                    'Reference': tc['reference'],
                    'Prediction': tc['prediction'],
                    'Exact_Match': tc['translation_metrics'].get('exact_match', 0.0),
                    'Word_Accuracy': tc['translation_metrics'].get('word_accuracy', 0.0),
                    'BLEU_4': tc['translation_metrics'].get('bleu_4', 0.0),
                    'Semantic_Similarity': tc['translation_metrics'].get('semantic_similarity', 0.0),
                    'LLM_Judge': tc['translation_metrics'].get('llm_judge_score', 0.0),
                    'Retrieval_Precision': tc['retrieval_metrics'].get('precision', 0.0)
                })
    print(f"✓ Summary CSV disimpan: {csv_file}")
    print(f"✓ Summary CSV terbaru disimpan: {latest_csv_file}")

    png_file = os.path.join(output_dir, f"summary_{timestamp}.png")
    save_evaluation_summary_png(results, png_file)
    latest_png_file = os.path.join(output_dir, "latest_summary.png")
    save_evaluation_summary_png(results, latest_png_file)
    print(f"✓ Summary PNG disimpan: {png_file}")
    print(f"✓ Summary PNG terbaru disimpan: {latest_png_file}")
    
    return results


def save_evaluation_summary_png(results: Dict, output_path: str) -> None:
    """Simpan ringkasan evaluasi satu arah sebagai grafik PNG untuk run saat ini."""
    overall = results.get('overall_metrics', {})

    metrics = {
        'BLEU-4': overall.get('bleu_4_mean', 0.0) * 100,
        'Semantic Similarity': overall.get('semantic_similarity_mean', 0.0) * 100,
        'LLM Judge Score': overall.get('llm_judge_score_mean', 0.0) * 20,
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    labels = list(metrics.keys())
    values = list(metrics.values())
    bars = ax.bar(labels, values, color=['#EF6C00', '#6A1B9A', '#1565C0'])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_title('Evaluation Summary (OCU -> ID)')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    """Fungsi utama evaluasi"""
    # Muat test set
    print("\nMemuat test set...")
    test_cases = load_test_set()
    print(f"✓ Loaded {len(test_cases)} test cases")
    
    # Inisialisasi translator
    print("\nInisialisasi translator...")
    try:
        translator = create_translator()
        print("✓ Translator berhasil diinisialisasi")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPastikan:")
        print("1. File .env sudah dikonfigurasi untuk OLLAMA_BASE_URL/LLM_MODEL")
        print("2. Vector store sudah dibuild (jalankan: python scripts/build_index.py)")
        sys.exit(1)
    
    # Jalankan evaluasi
    results = run_translation_evaluation(translator, test_cases)
    
    # Tampilkan ringkasan akhir
    print("\n\n" + "="*70)
    print("EVALUASI SELESAI".center(70))
    print("="*70)
    print(f"\n✓ Total test cases: {results['num_test_cases']}")
    print(f"✓ Overall Exact Match: {results['overall_metrics'].get('exact_match_mean', 0.0):.3f}")
    print(f"✓ Overall Word Accuracy: {results['overall_metrics'].get('word_accuracy_mean', 0.0):.3f}")
    print(f"✓ Overall BLEU-4: {results['overall_metrics']['bleu_4_mean']:.3f}")
    print(f"✓ Overall Semantic Similarity: {results['overall_metrics']['semantic_similarity_mean']:.3f}")
    print(f"✓ Overall LLM Judge: {results['overall_metrics'].get('llm_judge_score_mean', 0.0):.3f}/5")
    print(f"\nHasil evaluasi telah disimpan di folder 'evaluation_results/'\n")


if __name__ == "__main__":
    main()
