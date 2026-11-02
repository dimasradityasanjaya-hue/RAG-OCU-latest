"""Tahap 10 - Evaluasi Retrieval berbasis qrels basis pengetahuan."""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import load_test_set
from src.data_loader import TextPreprocessor
from src.retriever import RAGRetriever, RetrievalResult
from src.vector_store import load_vector_store


def load_qrels(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter=";")
        return {
            row["Test_ID"]: {
                "relevant_ids": {item for item in row["Relevant_KB_IDs"].split("|") if item},
                "status": row["Status"],
                "rationale": row["Rationale"],
            }
            for row in rows
        }


def infer_provenance_id(retriever: RAGRetriever, word: str, target: str) -> Optional[str]:
    """Resolve a synthetic single-word fallback to the KB row that produced it."""
    normalized_word = retriever.preprocessor.normalize_ocu(word)
    normalized_target = retriever.preprocessor.normalize(target)
    for entry in retriever.vector_store.entries:
        if entry.entry_type not in {"frasa", "kosakata"}:
            continue
        source_tokens = retriever.preprocessor.normalize_ocu(entry.bahasa_ocu).split()
        target_tokens = retriever.preprocessor.normalize(entry.bahasa_indonesia).split()
        if len(source_tokens) != len(target_tokens) or normalized_word not in source_tokens:
            continue
        index = source_tokens.index(normalized_word)
        if target_tokens[index] == normalized_target:
            return entry.id
    return None


def result_id(retriever: RAGRetriever, result: RetrievalResult) -> str:
    entry_id = result.entry.id
    if entry_id.startswith("fallback-"):
        resolved = infer_provenance_id(
            retriever, result.entry.bahasa_ocu, result.entry.bahasa_indonesia
        )
        return resolved or entry_id
    return entry_id


def ranked_context_ids(retriever: RAGRetriever, query: str, top_k: int) -> List[str]:
    """Return unique context IDs in the same priority order used in the prompt."""
    data = retriever.retrieve_with_word_breakdown(query, top_k=top_k)
    ranked: List[str] = []
    for results in data["word_breakdown"].values():
        for result in results:
            entry_id = result_id(retriever, result)
            if entry_id not in ranked:
                ranked.append(entry_id)
    for result in data["full_query"]:
        entry_id = result_id(retriever, result)
        if entry_id not in ranked:
            ranked.append(entry_id)
    return ranked[:top_k]


def metrics_at_k(ranked: List[str], relevant: set, top_k: int) -> dict:
    selected = ranked[:top_k]
    hits = [entry_id for entry_id in selected if entry_id in relevant]
    first_rank = next(
        (rank for rank, entry_id in enumerate(selected, start=1) if entry_id in relevant),
        None,
    )
    return {
        "precision_at_5": len(hits) / top_k,
        "recall_at_5": len(set(hits)) / len(relevant),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "hits": hits,
    }


def mean(items: List[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def save_chart(summary: dict, output_path: Path) -> None:
    labels = ["Kosakata", "Frasa", "Kalimat", "Keseluruhan"]
    keys = ["kosakata", "frasa", "kalimat", "overall"]
    metrics = [
        ("Precision@5", "precision_at_5", "#1565C0"),
        ("Recall@5", "recall_at_5", "#2E7D32"),
        ("MRR", "mrr", "#EF6C00"),
    ]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    for offset, (label, metric_key, color) in zip((-width, 0, width), metrics):
        values = [summary[key][metric_key] for key in keys]
        bars = ax.bar([value + offset for value in x], values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_title("Evaluasi Kinerja Retrieval Berbasis Qrels")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default="dataset/test_set.csv")
    parser.add_argument("--qrels-file", default="dataset/retrieval_qrels.csv")
    parser.add_argument("--output-dir", default="evaluation_results/test_set_translategemma4b/retrieval")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    tests = load_test_set(args.test_file)
    qrels = load_qrels(args.qrels_file)
    missing = [case["id"] for case in tests if case["id"] not in qrels]
    if missing:
        raise ValueError(f"Missing qrels for: {', '.join(missing)}")

    retriever = RAGRetriever(load_vector_store())
    details = []
    for case in tests:
        judgment = qrels[case["id"]]
        ranked = ranked_context_ids(retriever, case["input"], args.top_k)
        row = {
            "id": case["id"],
            "type": case["type"],
            "input": case["input"],
            "status": judgment["status"],
            "rationale": judgment["rationale"],
            "relevant_ids": sorted(judgment["relevant_ids"]),
            "retrieved_ids": ranked,
            "metrics": None,
        }
        if judgment["relevant_ids"]:
            row["metrics"] = metrics_at_k(ranked, judgment["relevant_ids"], args.top_k)
        details.append(row)

    def aggregate(rows: List[dict]) -> dict:
        measured = [row["metrics"] for row in rows if row["metrics"] is not None]
        return {
            "num_test_cases": len(rows),
            "num_evaluable": len(measured),
            "num_without_kb_evidence": len(rows) - len(measured),
            "precision_at_5": mean([item["precision_at_5"] for item in measured]),
            "recall_at_5": mean([item["recall_at_5"] for item in measured]),
            "mrr": mean([item["mrr"] for item in measured]),
        }

    summary = {
        entry_type: aggregate([row for row in details if row["type"] == entry_type])
        for entry_type in ("kosakata", "frasa", "kalimat")
    }
    summary["overall"] = aggregate(details)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "test_file": args.test_file,
        "qrels_file": args.qrels_file,
        "top_k": args.top_k,
        "aggregation_policy": "macro average over test cases with at least one relevant KB ID",
        "summary": summary,
        "details": details,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"retrieval_qrels_{timestamp}.json"
    csv_path = output_dir / f"retrieval_qrels_{timestamp}.csv"
    png_path = output_dir / f"retrieval_qrels_{timestamp}.png"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest_retrieval_qrels.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Type", "Status", "Relevant_IDs", "Retrieved_IDs", "Precision@5", "Recall@5", "MRR"])
        for row in details:
            metrics = row["metrics"] or {}
            writer.writerow([
                row["id"], row["type"], row["status"], "|".join(row["relevant_ids"]),
                "|".join(row["retrieved_ids"]), metrics.get("precision_at_5", ""),
                metrics.get("recall_at_5", ""), metrics.get("mrr", ""),
            ])
    save_chart(summary, png_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
