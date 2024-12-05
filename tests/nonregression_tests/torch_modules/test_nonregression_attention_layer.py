# tests/nonregression_tests/torch_modules/test_nonregression_attention_layer.py

import pytest

@pytest.mark.nonregression
def test_mocked() -> None:
    """Test mocked.

    Args:
        None

    Returns:
        None
    """
    assert True