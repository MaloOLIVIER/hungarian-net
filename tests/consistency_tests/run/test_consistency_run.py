# tests/consistency_tests/run/test_consistency_run.py

import pytest

@pytest.mark.consistency
def test_mocked() -> None:
    """Test mocked.

    Args:
        None

    Returns:
        None
    """
    assert True