import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_evaluation


class FakeTranslationEvaluator:
    def evaluate_single(self, prediction, reference, input_text):
        return {
            "bleu_4": 0.5,
            "semantic_similarity": 0.75,
            "llm_judge_score": 4.0,
        }

    def evaluate_batch(self, predictions, references, inputs=None):
        return ({
            "bleu_4_mean": 0.5,
            "semantic_similarity_mean": 0.75,
            "llm_judge_score_mean": 4.0,
        }, [])


class FakeRetrievalEvaluator:
    def evaluate_retrieval(self, **kwargs):
        return {"avg_similarity": 0.8, "precision": 1.0}


class FakeTranslator:
    def translate(self, input_text):
        return type("TranslationResult", (), {
            "translated_text": "terjemahan uji",
            "retrieval_results": [],
        })()


def one_test_case():
    return [{
        "id": "T1",
        "input": "ocu uji",
        "reference": "terjemahan uji",
        "type": "kalimat",
        "keterangan": "uji",
    }]


def configure_fake_evaluators(monkeypatch):
    monkeypatch.setattr(run_evaluation, "TranslationEvaluator", FakeTranslationEvaluator)
    monkeypatch.setattr(run_evaluation, "RetrievalEvaluator", FakeRetrievalEvaluator)


def snapshot_files(directory):
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.fixture(autouse=True)
def repository_evaluation_results_are_unchanged():
    evaluation_results = Path(__file__).parents[1] / "evaluation_results"
    before = snapshot_files(evaluation_results)
    yield
    assert snapshot_files(evaluation_results) == before


def test_run_metadata_is_copied_and_persisted_in_sorted_report(monkeypatch, tmp_path):
    configure_fake_evaluators(monkeypatch)
    metadata = {"zeta": 3, "alpha": "first", "nested": {"values": ["original"]}}

    results = run_evaluation.run_translation_evaluation(
        FakeTranslator(), one_test_case(), str(tmp_path), run_metadata=metadata
    )
    metadata["new_key"] = "mutated after evaluation"
    metadata["nested"]["values"].append("mutated after evaluation")

    expected_metadata = {"zeta": 3, "alpha": "first", "nested": {"values": ["original"]}}
    assert results["run_metadata"] == expected_metadata
    assert metadata == {
        "zeta": 3,
        "alpha": "first",
        "nested": {"values": ["original", "mutated after evaluation"]},
        "new_key": "mutated after evaluation",
    }

    stored_results = json.loads(next(tmp_path.glob("evaluation_*.json")).read_text(encoding="utf-8"))
    assert stored_results["run_metadata"] == expected_metadata

    report = next(tmp_path.glob("report_*.txt")).read_text(encoding="utf-8")
    assert "METADATA EKSPERIMEN" in report
    assert report.index("alpha") < report.index("zeta")
    assert "Exact Match" in report
    assert "Word Accuracy" in report

    with next(tmp_path.glob("summary_*.csv")).open(encoding="utf-8", newline="") as stream:
        assert next(csv.reader(stream)) == [
            "ID", "Type", "Input", "Reference", "Prediction", "Exact_Match",
            "Word_Accuracy", "BLEU_4", "Semantic_Similarity", "LLM_Judge",
            "Retrieval_Precision",
        ]


def test_legacy_call_without_run_metadata_returns_empty_dict_without_report_section(monkeypatch, tmp_path):
    configure_fake_evaluators(monkeypatch)

    results = run_evaluation.run_translation_evaluation(FakeTranslator(), one_test_case(), str(tmp_path))

    assert results["run_metadata"] == {}
    report = next(tmp_path.glob("report_*.txt")).read_text(encoding="utf-8")
    assert "METADATA EKSPERIMEN" not in report


def test_empty_metadata_returns_empty_dict_without_report_section(monkeypatch, tmp_path):
    configure_fake_evaluators(monkeypatch)

    results = run_evaluation.run_translation_evaluation(
        FakeTranslator(), one_test_case(), str(tmp_path), run_metadata={}
    )

    assert results["run_metadata"] == {}
    report = next(tmp_path.glob("report_*.txt")).read_text(encoding="utf-8")
    assert "METADATA EKSPERIMEN" not in report


def test_missing_retrieval_average_similarity_keeps_successful_case_and_writes_outputs(monkeypatch, tmp_path):
    class RetrievalEvaluatorWithoutAverage:
        def evaluate_retrieval(self, **kwargs):
            return {"precision": 1.0, "recall": 1.0, "mrr": 1.0}

    class TranslatorWithRetrievedItems:
        def translate(self, input_text):
            first_entry = type("Entry", (), {
                "bahasa_ocu": "ocu satu",
                "bahasa_indonesia": "indo satu",
                "entry_type": "kosakata",
            })()
            second_entry = type("Entry", (), {
                "bahasa_ocu": "ocu dua",
                "bahasa_indonesia": "indo dua",
                "entry_type": "kosakata",
            })()
            return type("TranslationResult", (), {
                "translated_text": "terjemahan uji",
                "retrieval_results": [
                    type("Retrieved", (), {"entry": first_entry, "score": 0.2})(),
                    type("Retrieved", (), {"entry": second_entry, "score": 0.8})(),
                ],
            })()

    monkeypatch.setattr(run_evaluation, "TranslationEvaluator", FakeTranslationEvaluator)
    monkeypatch.setattr(run_evaluation, "RetrievalEvaluator", RetrievalEvaluatorWithoutAverage)

    results = run_evaluation.run_translation_evaluation(
        TranslatorWithRetrievedItems(), one_test_case(), str(tmp_path)
    )

    assert results["test_cases"][0]["prediction"] == "terjemahan uji"
    assert results["test_cases"][0]["retrieval_metrics"]["avg_similarity"] == pytest.approx(0.5)
    assert len(list(tmp_path.glob("evaluation_*.json"))) == 1
    assert len(list(tmp_path.glob("report_*.txt"))) == 1
    assert len(list(tmp_path.glob("summary_*.csv"))) == 1
    assert len(list(tmp_path.glob("summary_*.png"))) == 1


def test_all_failed_cases_write_empty_overall_metrics_and_all_outputs(monkeypatch, tmp_path):
    class FailingTranslator:
        def translate(self, input_text):
            raise RuntimeError("translation failed")

    configure_fake_evaluators(monkeypatch)

    results = run_evaluation.run_translation_evaluation(
        FailingTranslator(), one_test_case(), str(tmp_path)
    )

    assert results["overall_metrics"] == {}
    assert results["test_cases"][0]["prediction"] is None
    assert results["test_cases"][0]["error"] == "translation failed"
    assert len(list(tmp_path.glob("evaluation_*.json"))) == 1
    assert len(list(tmp_path.glob("report_*.txt"))) == 1
    assert len(list(tmp_path.glob("summary_*.csv"))) == 1
    assert len(list(tmp_path.glob("summary_*.png"))) == 1


def test_latest_artifacts_use_stable_names(monkeypatch, tmp_path):
    configure_fake_evaluators(monkeypatch)

    run_evaluation.run_translation_evaluation(FakeTranslator(), one_test_case(), str(tmp_path))

    assert (tmp_path / "latest_evaluation.json").exists()
    assert (tmp_path / "latest_report.txt").exists()
    assert (tmp_path / "latest_summary.csv").exists()
    assert (tmp_path / "latest_summary.png").exists()


def test_unsuccessful_translation_result_records_error_without_metrics(monkeypatch, tmp_path):
    class UnsuccessfulTranslator:
        def translate(self, input_text):
            return type("TranslationResult", (), {
                "translated_text": "",
                "retrieval_results": [],
                "success": False,
                "error_message": "local model unavailable",
            })()

    configure_fake_evaluators(monkeypatch)

    results = run_evaluation.run_translation_evaluation(
        UnsuccessfulTranslator(), one_test_case(), str(tmp_path)
    )

    assert results["overall_metrics"] == {}
    assert results["metrics_by_type"] == {}
    assert results["test_cases"] == [{
        "id": "T1",
        "type": "kalimat",
        "input": "ocu uji",
        "reference": "terjemahan uji",
        "prediction": None,
        "error": "local model unavailable",
    }]
