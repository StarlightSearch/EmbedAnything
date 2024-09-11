from embed_anything import embed_directory, embed_file, embed_query


def test_openai_model_file(openai_model, test_pdf_file):
    data = embed_file(test_pdf_file, openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536

def test_openai_model_directory(openai_model, test_files_directory):
    data = embed_directory(test_files_directory, openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536

def test_openai_model_query(openai_model):
    data = embed_query(["Hello world"], openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536