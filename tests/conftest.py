"""
Test configuration.
Tests that require tensorflow are skipped when it is not installed.
"""
import sys
import pytest


def pytest_collection_modifyitems(items):
    """Skip tensorflow-dependent tests when tensorflow is not available."""
    try:
        import tensorflow  # noqa: F401
        tf_available = True
    except ImportError:
        tf_available = False

    if tf_available:
        return

    # Tests that directly import or require tensorflow
    TF_TEST_CLASSES = {
        "TestEmotionCNN",
        "TestAttention",
        "TestDecoder",
        "TestCaptionGenerator",
        "TestCOCOVocabulary",
    }
    TF_TEST_METHODS = {
        "test_forward_pass",
        "test_greedy_decode_produces_string",
    }

    skip_tf = pytest.mark.skip(reason="tensorflow not installed")
    for item in items:
        cls = item.cls.__name__ if item.cls else ""
        if cls in TF_TEST_CLASSES or item.name in TF_TEST_METHODS:
            item.add_marker(skip_tf)
