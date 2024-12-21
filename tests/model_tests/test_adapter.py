import embed_anything
import pytest


def test_adapter_upsert_call_file(
    bert_model, dummy_adapter, test_pdf_file, test_txt_file
):
    assert (
        embed_anything.embed_file(
            test_pdf_file, embedder=bert_model, adapter=dummy_adapter
        )
        is None
    )
    assert (
        embed_anything.embed_file(
            test_txt_file, embedder=bert_model, adapter=dummy_adapter
        )
        is None
    )


def test_adapter_upsert_call_directory(bert_model, dummy_adapter, test_files_directory):
    assert (
        embed_anything.embed_directory(
            test_files_directory, embedder=bert_model, adapter=dummy_adapter
        )
        is None
    )
