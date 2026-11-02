from src.data_loader import TextPreprocessor, TranslationEntry
from src.retriever import RAGRetriever


def test_single_word_query_uses_phrase_token_fallback():
    entries = [
        TranslationEntry("F1", "KAMPUONG GHAMAI", "Kampung ramai", "", "frasa"),
        TranslationEntry("F2", "KAMPUONG LAMO", "Kampung lama", "", "frasa"),
    ]
    store = type("Store", (), {"entries": entries})()
    retriever = RAGRetriever(store)

    mapping = retriever._infer_single_word_mapping("kampuong")

    assert mapping == {"kampuong": "kampung"}


def test_multiword_breakdown_does_not_leak_phrase_modifier():
    entries = [
        TranslationEntry("F1", "KAMPUONG GHAMAI", "Kampung ramai", "", "frasa"),
        TranslationEntry("F2", "KAMPUONG LAMO", "Kampung lama", "", "frasa"),
    ]

    class Store:
        def __init__(self):
            self.entries = entries

        def search(self, query, top_k=5):
            return []

    retriever = RAGRetriever(Store())
    breakdown = retriever.retrieve_with_word_breakdown("deyen poi ka kampuong")

    result = breakdown["word_breakdown"]["kampuong"][0]
    assert result.entry.bahasa_indonesia == "kampung"
    assert result.entry.bahasa_indonesia != "kampung lama"
