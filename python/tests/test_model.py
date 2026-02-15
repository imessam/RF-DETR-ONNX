import numpy as np
import pytest
from unittest.mock import patch
from modules.model import RFDETRModel

@pytest.fixture
def mock_ort_session():
    """Fixture to mock OnnxRuntimeSession."""
    with patch("modules.model.OnnxRuntimeSession") as mock:
        session_instance = mock.return_value
        session_instance.get_input_shape.return_value = [1, 3, 384, 384]
        yield session_instance

def test_model_initialization(mock_ort_session):
    """Test RFDETRModel initialization."""
    model = RFDETRModel("dummy_path.onnx", device="cpu")
    assert model.input_height == 384
    assert model.input_width == 384
    assert model.means.shape == (3, 1, 1)
    assert model.stds.shape == (3, 1, 1)

def test_preprocess(mock_ort_session):
    """Test image preprocessing."""
    model = RFDETRModel("dummy_path.onnx", device="cpu")
    # Create a dummy BGR image (H, W, C)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    preprocessed = model._preprocess(dummy_image)
    
    # Check shape: (batch, channels, height, width)
    assert preprocessed.shape == (1, 3, 384, 384)
    assert preprocessed.dtype == np.float32

def test_predict_flow(mock_ort_session):
    """Test full prediction flow with mocked session."""
    model = RFDETRModel("dummy_path.onnx", device="cpu")
    dummy_image = np.zeros((384, 384, 3), dtype=np.uint8)
    
    # Mock ORT session run output
    # RF-DETR output: [boxes, scores] or [boxes, scores, masks]
    mock_boxes = np.random.rand(1, 300, 4).astype(np.float32)
    mock_scores = np.random.rand(1, 300, 80).astype(np.float32)
    mock_ort_session.run.return_value = [mock_boxes, mock_scores]
    
    scores, labels, boxes, masks, timings = model.predict(dummy_image)
    
    assert isinstance(scores, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(boxes, np.ndarray)
    assert masks is None
    assert "preprocess" in timings
    assert "ort_run" in timings
    assert "postprocess" in timings
    assert "total" in timings

def test_post_process_shapes(mock_ort_session):
    """Test post-processing logic and output shapes."""
    model = RFDETRModel("dummy_path.onnx", device="cpu")
    
    # Mock data: 300 queries, 80 classes
    mock_boxes = np.zeros((1, 300, 4), dtype=np.float32)
    mock_scores = np.zeros((1, 300, 80), dtype=np.float32)
    
    # Set one detection
    mock_scores[0, 0, 5] = 10.0  # High score for class 5
    
    outputs = [mock_boxes, mock_scores]
    scores, labels, boxes, masks = model._post_process(outputs, 720, 1280, confidence_threshold=0.5)
    
    assert len(scores) > 0
    assert labels[0] == 5
    assert boxes.shape[1] == 4
    # Check that boxes are scaled to 1280x720
    assert boxes[0, 2] <= 1280
    assert boxes[0, 3] <= 720
