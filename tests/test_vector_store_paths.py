import hashlib

import numpy as np
import pytest

import src.vector_store as vector_store_module
from src.data_loader import TranslationEntry
from src.vector_store import build_vector_store, load_vector_store


class FakeEmbedding:
    def encode_documents(self, texts):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % 384] = 1.0
        return vectors

    def encode_query(self, text):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


def entry(identifier="K1"):
    return TranslationEntry(identifier, "ocu", "indonesia", "catatan", "kosakata")


def artifact_snapshot(paths):
    return {
        path: path.read_bytes() if path.exists() else None
        for path in paths
    }


def test_custom_paths_build_and_reload_without_changing_default_artifacts(tmp_path):
    default_index = vector_store_module.FAISS_INDEX_FILE
    default_metadata = vector_store_module.METADATA_FILE
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (default_index, default_metadata)
        if path.exists()
    }
    index_path = tmp_path / "indexes" / "holdout.index"
    metadata_path = tmp_path / "metadata" / "holdout.pkl"

    store = build_vector_store([entry()], index_path, metadata_path, FakeEmbedding())
    loaded = load_vector_store(index_path, metadata_path, FakeEmbedding(), build_if_missing=False)

    assert index_path.exists()
    assert metadata_path.exists()
    assert store.index.ntotal == len(store.metadata) == 1
    assert loaded.index.ntotal == len(loaded.metadata) == 1
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (default_index, default_metadata)
        if path.exists()
    }
    assert after == before


def test_missing_custom_paths_do_not_build_when_building_is_disabled(tmp_path, monkeypatch):
    paths = (tmp_path / "missing" / "index.faiss", tmp_path / "other" / "metadata.pkl")
    monkeypatch.setattr(vector_store_module, "build_vector_store", lambda: pytest.fail("must not build"))

    with pytest.raises(FileNotFoundError) as error:
        load_vector_store(*paths, embedding_model=FakeEmbedding(), build_if_missing=False)

    assert str(paths[0]) in str(error.value)
    assert str(paths[1]) in str(error.value)


def test_explicit_empty_entries_are_not_replaced_by_default_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module.DataLoader, "load_all", lambda self: pytest.fail("must not load defaults"))

    with pytest.raises(ValueError, match="empty"):
        build_vector_store([], tmp_path / "empty.index", tmp_path / "empty.pkl", FakeEmbedding())


@pytest.mark.parametrize("path_keyword", ["index_path", "metadata_path"])
def test_build_rejects_a_single_explicit_path_without_changing_defaults(tmp_path, monkeypatch, path_keyword):
    default_paths = (tmp_path / "default.index", tmp_path / "default.pkl")
    for path, content in zip(default_paths, (b"default-index", b"default-metadata")):
        path.write_bytes(content)
    monkeypatch.setattr(vector_store_module, "FAISS_INDEX_FILE", default_paths[0])
    monkeypatch.setattr(vector_store_module, "METADATA_FILE", default_paths[1])
    custom_path = tmp_path / "experiment" / path_keyword
    before = artifact_snapshot(default_paths)

    with pytest.raises(ValueError, match="both index_path and metadata_path"):
        build_vector_store([entry()], embedding_model=FakeEmbedding(), **{path_keyword: custom_path})

    assert artifact_snapshot(default_paths) == before
    assert not custom_path.exists()


@pytest.mark.parametrize("path_keyword", ["index_path", "metadata_path"])
def test_load_rejects_a_single_explicit_path_before_constructing_store(tmp_path, monkeypatch, path_keyword):
    default_paths = (tmp_path / "default.index", tmp_path / "default.pkl")
    for path, content in zip(default_paths, (b"default-index", b"default-metadata")):
        path.write_bytes(content)
    monkeypatch.setattr(vector_store_module, "FAISS_INDEX_FILE", default_paths[0])
    monkeypatch.setattr(vector_store_module, "METADATA_FILE", default_paths[1])
    monkeypatch.setattr(
        vector_store_module,
        "VectorStore",
        lambda *args, **kwargs: pytest.fail("must not construct a store"),
    )
    custom_path = tmp_path / "experiment" / path_keyword
    before = artifact_snapshot(default_paths)

    with pytest.raises(ValueError, match="both index_path and metadata_path"):
        load_vector_store(embedding_model=FakeEmbedding(), **{path_keyword: custom_path})

    assert artifact_snapshot(default_paths) == before
    assert not custom_path.exists()


def test_existing_corrupt_custom_files_are_not_reported_as_missing(tmp_path):
    index_path = tmp_path / "corrupt.index"
    metadata_path = tmp_path / "corrupt.pkl"
    index_path.write_bytes(b"not a faiss index")
    metadata_path.write_bytes(b"not pickle metadata")

    with pytest.raises(ValueError, match="Unable to load vector store"):
        load_vector_store(index_path, metadata_path, FakeEmbedding(), build_if_missing=False)


def test_build_index_preserves_entries_from_an_iterator():
    store = vector_store_module.VectorStore(embedding_model=FakeEmbedding())

    store.build_index(iter([entry()]))

    assert len(store.entries) == 1
    assert len(store.metadata) == 1
    assert store.index.ntotal == 1


def test_search_rejects_invalid_input(tmp_path):
    store = build_vector_store(
        [entry()],
        tmp_path / "search.index",
        tmp_path / "search.pkl",
        embedding_model=FakeEmbedding(),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        store.search("   ")
    with pytest.raises(ValueError, match="positive integer"):
        store.search("ocu", top_k=0)


def test_search_rejects_index_metadata_mismatch(tmp_path):
    store = build_vector_store(
        [entry()],
        tmp_path / "mismatch.index",
        tmp_path / "mismatch.pkl",
        embedding_model=FakeEmbedding(),
    )
    store.entries.clear()

    with pytest.raises(ValueError, match="out of sync"):
        store.search("ocu")
