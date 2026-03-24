"""Pytest configuration and fixtures for tenax-qtt."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by file name, mirroring tenax conventions."""
    for item in items:
        path = str(item.fspath)
        if "test_grid" in path or "test_qtt" in path or "test_folding" in path:
            item.add_marker(pytest.mark.core)
        elif "test_cross" in path or "test_fourier" in path or "test_matrix" in path:
            item.add_marker(pytest.mark.algorithm)
        elif "test_arithmetic" in path:
            item.add_marker(pytest.mark.core)
