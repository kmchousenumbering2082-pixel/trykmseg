"""
Shared pytest fixtures for RALIS testing infrastructure.

This module provides common fixtures that can be used across all test files.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    return np.random.rand(10, 10, 3)


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing data loading functionality."""
    mock_data = MagicMock()
    mock_data.__len__ = MagicMock(return_value=100)
    mock_data.__getitem__ = MagicMock(return_value=(
        np.random.rand(3, 224, 224),  # sample image
        np.random.randint(0, 2, (224, 224))  # sample mask
    ))
    return mock_data


@pytest.fixture
def mock_model():
    """Mock model for testing model-related functionality."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.train = MagicMock()
    mock_model.parameters = MagicMock(return_value=[])
    return mock_model


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        'batch_size': 4,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'dataset': 'camvid',
        'model': 'fpn',
        'device': 'cpu'
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing logging functionality."""
    return MagicMock()


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images for testing."""
    return np.random.rand(4, 3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_mask_batch():
    """Create a sample batch of segmentation masks for testing."""
    return np.random.randint(0, 2, (4, 224, 224)).astype(np.int64)


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        yield 'cpu'


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Setup test environment with temporary directories and environment variables."""
    # Set up temporary directories
    test_data_dir = temp_dir / "test_data"
    test_output_dir = temp_dir / "test_output"
    test_data_dir.mkdir()
    test_output_dir.mkdir()
    
    # Set environment variables
    monkeypatch.setenv("TEST_DATA_DIR", str(test_data_dir))
    monkeypatch.setenv("TEST_OUTPUT_DIR", str(test_output_dir))
    
    return {
        "data_dir": test_data_dir,
        "output_dir": test_output_dir
    }