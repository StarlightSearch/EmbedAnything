from embed_anything import embed_directory, embed_file, embed_query


def test_openai_model_file(openai_model):
    data = embed_file("test_files/test.pdf", openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536

def test_openai_model_directory(openai_model):
    data = embed_directory("test_files", openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536

def test_openai_model_query(openai_model):
    data = embed_query(["Hello world"], openai_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536