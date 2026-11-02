"""
Tahap 10 - Perbandingan LLM-only vs RAG+LLM.

Skrip ini mengevaluasi dua mode pada input yang sama:
1. LLM only  -> terjemahan langsung tanpa konteks RAG
2. RAG+LLM   -> pipeline penerjemah dengan konteks retrieval
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import TranslationEvaluator, print_metrics
from src.llm import LLMTranslator
from src.translator import create_translator


ACTIVE_METRIC_KEYS = [
    'bleu_4_mean',
    'semantic_similarity_mean',
    'llm_judge_score_mean'
]


def load_test_set(test_file: str = "dataset/test_set.csv") -> List[Dict]:
    """Load Ocu -> Indonesian evaluation set."""
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


def summarize_metrics(
    evaluator: TranslationEvaluator,
    details: List[Dict],
    test_cases: List[Dict]
) -> Tuple[Dict, Dict]:
    """Aggregate overall metrics and metrics by type from per-case details."""
    predictions = []
    references = []
    inputs = []

    for detail in details:
        if detail.get('prediction') is None:
            continue
        predictions.append(detail['prediction'])
        references.append(detail['reference'])
        inputs.append(detail['input'])

    overall_metrics = {}
    if predictions:
        overall_metrics, _ = evaluator.evaluate_batch(predictions, references, inputs)

    metrics_by_type = {}
    for test_type in sorted({tc['type'] for tc in test_cases}):
        type_predictions = []
        type_references = []

        for detail in details:
            if detail['type'] != test_type or detail.get('prediction') is None:
                continue
            type_predictions.append(detail['prediction'])
            type_references.append(detail['reference'])

        if type_predictions:
            type_metrics, _ = evaluator.evaluate_batch(type_predictions, type_references)
            metrics_by_type[test_type] = type_metrics

    return overall_metrics, metrics_by_type


def evaluate_mode(
    mode_name: str,
    test_cases: List[Dict],
    evaluator: TranslationEvaluator,
    rag_translator=None,
    llm_only=None
) -> Dict:
    """Evaluate one translation mode against the same test set."""
    details = []

    print("\n" + "=" * 78)
    print(f"EVALUASI MODE: {mode_name}".center(78))
    print("=" * 78)

    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        reference = test_case['reference']

        print(f"\n[{i}/{len(test_cases)}] {test_case['id']} ({test_case['type']})")
        print(f"Input    : {input_text}")
        print(f"Reference: {reference}")

        try:
            if mode_name == "LLM Only":
                prediction = llm_only.translate_simple(input_text)
            else:
                result = rag_translator.translate(input_text)
                prediction = result.translated_text

            metrics = evaluator.evaluate_single(prediction, reference, input_text)

            print(f"Prediksi : {prediction}")
            print(f"  ✓ BLEU-4: {metrics['bleu_4']:.3f}")
            print(f"  ✓ Semantic Sim: {metrics['semantic_similarity']:.3f}")
            print(f"  ✓ LLM Judge: {metrics['llm_judge_score']:.1f}/5")

            details.append({
                'id': test_case['id'],
                'type': test_case['type'],
                'input': input_text,
                'reference': reference,
                'prediction': prediction,
                'metrics': metrics,
                'error': None
            })
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            details.append({
                'id': test_case['id'],
                'type': test_case['type'],
                'input': input_text,
                'reference': reference,
                'prediction': None,
                'metrics': {},
                'error': str(e)
            })

    overall_metrics, metrics_by_type = summarize_metrics(evaluator, details, test_cases)

    if overall_metrics:
        print_metrics(overall_metrics, f"METRIK KESELURUHAN - {mode_name}")

    return {
        'mode': mode_name,
        'overall_metrics': overall_metrics,
        'metrics_by_type': metrics_by_type,
        'details': details
    }


def build_comparison_summary(llm_only_results: Dict, rag_results: Dict) -> Dict:
    """Build compact metric deltas to show RAG impact over LLM-only."""
    metric_keys = ACTIVE_METRIC_KEYS

    overall_delta = {}
    for key in metric_keys:
        overall_delta[key] = (
            rag_results['overall_metrics'].get(key, 0.0)
            - llm_only_results['overall_metrics'].get(key, 0.0)
        )

    by_type_delta = {}
    all_types = set(llm_only_results['metrics_by_type']) | set(rag_results['metrics_by_type'])
    for test_type in sorted(all_types):
        by_type_delta[test_type] = {}
        llm_metrics = llm_only_results['metrics_by_type'].get(test_type, {})
        rag_metrics = rag_results['metrics_by_type'].get(test_type, {})
        for key in metric_keys:
            by_type_delta[test_type][key] = rag_metrics.get(key, 0.0) - llm_metrics.get(key, 0.0)

    return {
        'overall_delta_rag_minus_llm': overall_delta,
        'by_type_delta_rag_minus_llm': by_type_delta
    }


def save_comparison_png(results: Dict, output_path: str) -> None:
    """Save bar chart comparing key overall metrics for both modes."""
    llm_overall = results['modes']['llm_only']['overall_metrics']
    rag_overall = results['modes']['rag_llm']['overall_metrics']

    metrics = [
        ('BLEU-4', llm_overall.get('bleu_4_mean', 0.0) * 100, rag_overall.get('bleu_4_mean', 0.0) * 100),
        ('Semantic Similarity', llm_overall.get('semantic_similarity_mean', 0.0) * 100, rag_overall.get('semantic_similarity_mean', 0.0) * 100),
        ('LLM Judge Score', llm_overall.get('llm_judge_score_mean', 0.0) * 20, rag_overall.get('llm_judge_score_mean', 0.0) * 20),
    ]

    labels = [m[0] for m in metrics]
    llm_vals = [m[1] for m in metrics]
    rag_vals = [m[2] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    x = list(range(len(labels)))
    width = 0.36

    bars1 = ax.bar([i - width / 2 for i in x], llm_vals, width=width, label='LLM Only', color='#6A1B9A')
    bars2 = ax.bar([i + width / 2 for i in x], rag_vals, width=width, label='RAG + LLM', color='#1565C0')

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_title('LLM Only vs RAG + LLM')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_outputs(output_dir: str, results: Dict) -> Dict[str, str]:
    """Persist detailed comparison results as JSON, TXT, CSV, and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"evaluation_llm_vs_rag_{timestamp}.json")
    txt_path = os.path.join(output_dir, f"report_llm_vs_rag_{timestamp}.txt")
    csv_path = os.path.join(output_dir, f"summary_llm_vs_rag_{timestamp}.csv")
    png_path = os.path.join(output_dir, f"summary_llm_vs_rag_{timestamp}.png")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    llm_overall = results['modes']['llm_only']['overall_metrics']
    rag_overall = results['modes']['rag_llm']['overall_metrics']
    delta = results['comparison']['overall_delta_rag_minus_llm']

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 78 + "\n")
        f.write("LAPORAN PERBANDINGAN LLM ONLY VS RAG + LLM\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Tanggal: {results['timestamp']}\n")
        f.write(f"Jumlah Kasus: {results['num_test_cases']}\n\n")

        f.write("[LLM ONLY]\n")
        f.write("-" * 78 + "\n")
        for k, v in sorted(llm_overall.items()):
            if isinstance(v, float):
                f.write(f"{k:.<52} {v:.4f}\n")

        f.write("\n[RAG + LLM]\n")
        f.write("-" * 78 + "\n")
        for k, v in sorted(rag_overall.items()):
            if isinstance(v, float):
                f.write(f"{k:.<52} {v:.4f}\n")

        f.write("\n[DELTA RAG - LLM]\n")
        f.write("-" * 78 + "\n")
        for k, v in sorted(delta.items()):
            f.write(f"{k:.<52} {v:+.4f}\n")

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'ID', 'Type', 'Input', 'Reference',
            'LLM_Only_Prediction', 'RAG_LLM_Prediction',
            'LLM_Only_BLEU_4', 'RAG_LLM_BLEU_4',
            'LLM_Only_LLM_Judge', 'RAG_LLM_LLM_Judge',
            'LLM_Only_Semantic_Similarity', 'RAG_LLM_Semantic_Similarity'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        llm_details = {d['id']: d for d in results['modes']['llm_only']['details']}
        rag_details = {d['id']: d for d in results['modes']['rag_llm']['details']}

        for test_id in sorted(llm_details.keys() | rag_details.keys()):
            llm = llm_details.get(test_id, {})
            rag = rag_details.get(test_id, {})
            llm_metrics = llm.get('metrics', {})
            rag_metrics = rag.get('metrics', {})

            writer.writerow({
                'ID': test_id,
                'Type': llm.get('type') or rag.get('type'),
                'Input': llm.get('input') or rag.get('input'),
                'Reference': llm.get('reference') or rag.get('reference'),
                'LLM_Only_Prediction': llm.get('prediction'),
                'RAG_LLM_Prediction': rag.get('prediction'),
                'LLM_Only_BLEU_4': llm_metrics.get('bleu_4'),
                'RAG_LLM_BLEU_4': rag_metrics.get('bleu_4'),
                'LLM_Only_LLM_Judge': llm_metrics.get('llm_judge_score'),
                'RAG_LLM_LLM_Judge': rag_metrics.get('llm_judge_score'),
                'LLM_Only_Semantic_Similarity': llm_metrics.get('semantic_similarity'),
                'RAG_LLM_Semantic_Similarity': rag_metrics.get('semantic_similarity')
            })

    save_comparison_png(results, png_path)

    return {
        'json': json_path,
        'txt': txt_path,
        'csv': csv_path,
        'png': png_path
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-only vs RAG+LLM on Ocu test set")
    parser.add_argument(
        "--test-file",
        default="dataset/test_set.csv",
        help="Path ke test set CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Folder output hasil evaluasi"
    )
    args = parser.parse_args()

    print("\nMemuat test set...")
    test_cases = load_test_set(args.test_file)
    print(f"✓ Loaded {len(test_cases)} test cases")

    print("\nInisialisasi LLM-only baseline...")
    llm_only = LLMTranslator()

    print("\nInisialisasi translator RAG+LLM...")
    rag_translator = create_translator()

    print("\nInisialisasi evaluator...")
    shared_embedding_model = rag_translator.retriever.vector_store.embedding_model.model
    evaluator = TranslationEvaluator(embedding_model=shared_embedding_model)

    llm_only_results = evaluate_mode(
        mode_name="LLM Only",
        test_cases=test_cases,
        evaluator=evaluator,
        llm_only=llm_only
    )
    rag_results = evaluate_mode(
        mode_name="RAG + LLM",
        test_cases=test_cases,
        evaluator=evaluator,
        rag_translator=rag_translator
    )

    results = {
        'timestamp': datetime.now().isoformat(),
        'num_test_cases': len(test_cases),
        'test_file': args.test_file,
        'modes': {
            'llm_only': llm_only_results,
            'rag_llm': rag_results
        },
        'comparison': build_comparison_summary(llm_only_results, rag_results)
    }

    print_metrics(llm_only_results['overall_metrics'], "METRIK KESELURUHAN - LLM ONLY")
    print_metrics(rag_results['overall_metrics'], "METRIK KESELURUHAN - RAG + LLM")

    print("\n" + "=" * 78)
    print("DELTA RAG - LLM".center(78))
    print("=" * 78)
    for key, value in sorted(results['comparison']['overall_delta_rag_minus_llm'].items()):
        print(f"{key:.<52} {value:+.4f}")

    paths = save_outputs(args.output_dir, results)

    print("\n" + "=" * 78)
    print("EVALUASI LLM ONLY VS RAG + LLM SELESAI".center(78))
    print("=" * 78)
    print(f"✓ JSON : {paths['json']}")
    print(f"✓ TXT  : {paths['txt']}")
    print(f"✓ CSV  : {paths['csv']}")
    print(f"✓ PNG  : {paths['png']}")


if __name__ == "__main__":
    main()
