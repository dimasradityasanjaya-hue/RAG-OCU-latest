"""Run dataset-type ablation on the fixed Ocu test set."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import load_test_set, run_translation_evaluation
from src.config import EMBEDDING_MODEL, LLM_MODEL
from src.translator import create_translator
from src.vector_store import VectorStore, load_vector_store


CONFIGURATIONS = {
    "baseline": None,
    "tanpa_kosakata": "kosakata",
    "tanpa_frasa": "frasa",
    "tanpa_kalimat": "kalimat",
}


def subset_store(full_store: VectorStore, excluded_type: str) -> VectorStore:
    """Create an isolated FAISS store by copying vectors from the full index."""
    selected_indices = [
        index for index, entry in enumerate(full_store.entries)
        if entry.entry_type != excluded_type
    ]
    selected_entries = [full_store.entries[index] for index in selected_indices]
    vectors = np.vstack([full_store.index.reconstruct(index) for index in selected_indices]).astype(np.float32)

    store = VectorStore(embedding_model=full_store.embedding_model)
    store.entries = selected_entries
    store.metadata = [entry.to_dict() for entry in selected_entries]
    store.index = faiss.IndexFlatIP(vectors.shape[1])
    store.index.add(vectors)
    return store


def metric_snapshot(result: dict) -> dict:
    metrics = result.get("overall_metrics", {})
    return {
        "num_entries": result.get("run_metadata", {}).get("knowledge_base_entries", 0),
        "exact_match": metrics.get("exact_match_mean", 0.0),
        "word_accuracy": metrics.get("word_accuracy_mean", 0.0),
        "bleu_4": metrics.get("bleu_4_mean", 0.0),
        "semantic_similarity": metrics.get("semantic_similarity_mean", 0.0),
        "llm_judge": metrics.get("llm_judge_score_mean", 0.0),
    }


def save_chart(summary: dict, path: Path) -> None:
    labels = ["Baseline", "Tanpa\nKosakata", "Tanpa\nFrasa", "Tanpa\nKalimat"]
    keys = list(CONFIGURATIONS)
    metric_defs = [
        ("Exact Match", "exact_match"),
        ("Word Accuracy", "word_accuracy"),
        ("BLEU-4", "bleu_4"),
        ("Semantic Similarity", "semantic_similarity"),
    ]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    for index, (label, metric) in enumerate(metric_defs):
        offset = (index - 1.5) * width
        values = [summary[key][metric] for key in keys]
        bars = ax.bar(x + offset, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_title("Ablation Study Kontribusi Dataset")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default="dataset/test_set.csv")
    parser.add_argument(
        "--baseline-file",
        default="evaluation_results/latest_evaluation.json",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/test_set_translategemma4b/ablation",
    )
    args = parser.parse_args()

    tests = load_test_set(args.test_file)
    full_store = load_vector_store()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.baseline_file, encoding="utf-8") as stream:
        results = {"baseline": json.load(stream)}
    baseline_metadata = dict(results["baseline"].get("run_metadata", {}))
    # Selalu gunakan jumlah entri dari vector store aktif agar metadata
    # baseline tidak mengambil angka lama dari berkas evaluasi sebelumnya.
    baseline_metadata["knowledge_base_entries"] = len(full_store.entries)
    results["baseline"]["run_metadata"] = baseline_metadata
    # Evaluasi utama menyimpan nama model sebagai `model`, sedangkan
    # beberapa hasil lama menggunakan `llm_model`. Terima keduanya.
    # Baseline terbaru dapat tidak menyimpan metadata model. Dalam kondisi
    # tersebut, gunakan model aktif; metadata model yang berbeda tetap ditolak.
    baseline_model = (
        baseline_metadata.get("llm_model")
        or baseline_metadata.get("model")
        or LLM_MODEL
    )
    if baseline_model != LLM_MODEL:
        raise ValueError("Baseline model does not match the active LLM model")
    baseline_metadata["llm_model"] = baseline_model
    if results["baseline"].get("num_test_cases") != len(tests):
        raise ValueError("Baseline test count does not match the active test set")

    for name, excluded_type in CONFIGURATIONS.items():
        if name == "baseline":
            continue
        store = subset_store(full_store, excluded_type)
        translator = create_translator(vector_store=store)
        metadata = {
            "test_file": args.test_file,
            "experiment": "dataset_type_ablation",
            "configuration": name,
            "excluded_entry_type": excluded_type,
            "knowledge_base_entries": len(store.entries),
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "vector_source": "reconstructed from the fixed full FAISS index",
        }
        results[name] = run_translation_evaluation(
            translator,
            tests,
            output_dir=str(output_dir / name),
            run_metadata=metadata,
        )

    summary = {name: metric_snapshot(results[name]) for name in CONFIGURATIONS}
    payload = {
        "timestamp": datetime.now().isoformat(),
        "test_file": args.test_file,
        "llm_model": LLM_MODEL,
        "baseline_file": args.baseline_file,
        "note": "The baseline is reused from the validated full RAG evaluation; other configurations use copied vectors and identical runtime settings.",
        "summary": summary,
        "results": results,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ablation_{timestamp}.json"
    png_path = output_dir / f"ablation_{timestamp}.png"
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    (output_dir / "latest_ablation.json").write_text(json_text, encoding="utf-8")
    save_chart(summary, png_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
