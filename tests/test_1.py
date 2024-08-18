import pytest
import embed_anything
from embed_anything import EmbedData
# from embed_anything import embed_query

# test_with_pytest.py

def test_embed_query_with_empty_string():
    result = embed_anything.embed_query([''], embeder='Clip')
    # Assuming the function returns a list with an EmbedData object for an empty string
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], EmbedData)


# def test_embed_query_with_single_word():
#     result = embed_query('hello')
#     assert result == '<embed>hello</embed>'

# def test_embed_query_with_multiple_words():
#     result = embed_query('hello world')
#     assert result == '<embed>hello world</embed>'

# def test_embed_query_with_special_characters():
#     result = embed_query('!@#$%^&*()')
#     assert result == '<embed>!@#$%^&*()</embed>'

# def test_embed_query_with_numbers():
#     result = embed_query('12345')
#     assert result == '<embed>12345</embed>'

# def test_embed_query_with_whitespace():
#     result = embed_query('   hello   ')
#     assert result == '<embed>   hello   </embed>'

# def test_always_fails():
#     assert False