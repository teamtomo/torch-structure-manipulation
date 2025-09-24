import torch_structure_manipulation


def test_imports_with_version():
    assert isinstance(torch_structure_manipulation.__version__, str)
