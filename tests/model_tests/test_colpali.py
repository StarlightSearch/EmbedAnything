import pytest
from embed_anything import ColpaliModel


@pytest.mark.parametrize("model_fixture", ["colpali_model", "colpali_onnx_model"])
def test_colpali_model_file(model_fixture, test_pdf_file, request):
    model: ColpaliModel = request.getfixturevalue(model_fixture)
    data = model.embed_file(test_pdf_file, batch_size=1)
    assert len(data) == 1
