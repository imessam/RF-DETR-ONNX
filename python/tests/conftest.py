import pytest

def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption("--arch", action="store", default="nano", help="Torch model architecture (nano, small, base, medium)")
    parser.addoption("--device", action="store", default="gpu", choices=["cpu", "gpu"], help="Target device (cpu, gpu)")
    parser.addoption("--model_path", action="store", default="tests/test_models/inference_model.sim.onnx", help="Path to ONNX model")
    parser.addoption("--threshold", action="store", default=0.5, type=float, help="Confidence threshold")
    parser.addoption("--assets_dir", action="store", default="tests/test_assets", help="Directory containing test assets")
    parser.addoption("--results_dir", action="store", default="tests/test_results", help="Directory to save test results")

@pytest.fixture(scope="session")
def test_config(request):
    """Fixture to provide test configuration from command line options."""
    return {
        "arch": request.config.getoption("--arch"),
        "device": request.config.getoption("--device"),
        "model_path": request.config.getoption("--model_path"),
        "threshold": request.config.getoption("--threshold"),
        "assets_dir": request.config.getoption("--assets_dir"),
        "results_dir": request.config.getoption("--results_dir")
    }
