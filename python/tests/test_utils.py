import numpy as np
import pytest
import os
import sys

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from modules.utils import sigmoid, box_cxcywh_to_xyxyn

def test_sigmoid():
    x = np.array([-100.0, 0.0, 100.0])
    expected = 1 / (1 + np.exp(-x))
    output = sigmoid(x)
    
    np.testing.assert_allclose(output, expected, atol=1e-6)
    assert output[0] < 0.01
    assert abs(output[1] - 0.5) < 1e-6
    assert output[2] > 0.99

def test_box_cxcywh_to_xyxyn():
    # Input: cx, cy, w, h
    x = np.array([[0.5, 0.5, 0.2, 0.4]])
    
    # Expected: xmin = 0.5 - 0.1 = 0.4
    #           ymin = 0.5 - 0.2 = 0.3
    #           xmax = 0.5 + 0.1 = 0.6
    #           ymax = 0.5 + 0.2 = 0.7
    expected = np.array([[0.4, 0.3, 0.6, 0.7]])
    output = box_cxcywh_to_xyxyn(x)
    
    np.testing.assert_allclose(output, expected, atol=1e-6)
