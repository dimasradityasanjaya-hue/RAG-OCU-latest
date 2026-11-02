import pytest

from src.data_loader import TextPreprocessor, TranslationEntry
from src.retriever import RetrievalResult
from src.translator import OcuTranslator, create_translator


def test_translator_uses_injected_store_without_loading_or_building(monkeypatch):
    supplied_store = object()
    captured = {}

    monkeypatch.setattr("src.translator.load_vector_store", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    monkeypatch.setattr("src.translator.build_vector_store", lambda: (_ for _ in ()).throw(AssertionError("must not build")))
    monkeypatch.setattr("src.translator.RAGRetriever", lambda store: captured.setdefault("store", store))
    monkeypatch.setattr("src.translator.LLMTranslator", lambda api_key: object())

    OcuTranslator(vector_store=supplied_store, rebuild_index=True)

    assert captured["store"] is supplied_store


def test_factory_forwards_injected_store(monkeypatch):
    supplied_store = object()
    captured = {}

    monkeypatch.setattr("src.translator.OcuTranslator", lambda **kwargs: captured.setdefault("kwargs", kwargs))

    create_translator(api_key="key", rebuild_index=True, vector_store=supplied_store)

    assert captured["kwargs"] == {
        "api_key": "key",
        "rebuild_index": True,
        "vector_store": supplied_store,
    }


def test_existing_default_call_signatures_remain_callable(monkeypatch):
    monkeypatch.setattr("src.translator.OcuTranslator", lambda **kwargs: kwargs)

    assert create_translator() == {"api_key": None, "rebuild_index": False, "vector_store": None}


@pytest.mark.parametrize(
    ("input_text", "stored_ocu", "stored_meanings"),
    [
        ("uwuo", "uwuo", "kakak laki-laki / kakak perempuan"),
        ("CINAN", "cinan", "ingin / mau"),
        ("coga", "coga", "ganteng / cantik"),
        ("omak", "omak", "ibu / emak"),
    ],
)
def test_exact_ambiguous_kosakata_returns_original_text(
    monkeypatch, input_text, stored_ocu, stored_meanings
):
    ambiguous_entry = TranslationEntry(
        id="K1",
        bahasa_ocu=stored_ocu,
        bahasa_indonesia=stored_meanings,
        keterangan="Nomina",
        entry_type="kosakata",
    )

    supplied_store = type("Store", (), {"entries": [ambiguous_entry]})()

    class FakeRetriever:
        def __init__(self, store):
            self.vector_store = store

        def retrieve_with_word_breakdown(self, query):
            return {
                "full_query": [RetrievalResult(entry=ambiguous_entry, score=1.0)],
                "word_breakdown": {},
            }

        def format_context_for_llm(self, **kwargs):
            return "ctx"

    class FakeLLM:
        def translate(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for ambiguous exact matches")

    monkeypatch.setattr("src.translator.load_vector_store", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    monkeypatch.setattr("src.translator.build_vector_store", lambda: (_ for _ in ()).throw(AssertionError("must not build")))
    monkeypatch.setattr("src.translator.RAGRetriever", FakeRetriever)
    monkeypatch.setattr("src.translator.LLMTranslator", lambda api_key: FakeLLM())

    translator = OcuTranslator(vector_store=supplied_store)
    result = translator.translate(input_text)

    assert result.success is True
    assert result.translated_text == input_text


def test_distinct_meanings_across_kosakata_entries_are_ambiguous():
    entries = [
        TranslationEntry("K1", "contoh", "arti pertama", "", "kosakata"),
        TranslationEntry("K2", "contoh", "arti kedua", "", "kosakata"),
    ]
    translator = OcuTranslator.__new__(OcuTranslator)
    translator.preprocessor = TextPreprocessor()
    translator.retriever = type(
        "Retriever", (), {"vector_store": type("Store", (), {"entries": entries})()}
    )()
    translator._ambiguity_cache = None

    assert translator._has_ambiguous_exact_source("contoh") is True
