import pytest

def pytest_addoption(parser):
    parser.addoption("--reference", action="store", help="Path to reference results directory")
    parser.addoption("--python_onnx", action="store", help="Path to python onnx results directory")
    parser.addoption("--cpp_onnx", action="store", help="Path to cpp onnx results directory")

@pytest.fixture
def ref_dir(request):
    return request.config.getoption("--reference")

@pytest.fixture
def py_dir(request):
    return request.config.getoption("--python_onnx")

@pytest.fixture
def cpp_dir(request):
    return request.config.getoption("--cpp_onnx")
