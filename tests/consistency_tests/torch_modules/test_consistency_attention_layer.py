# tests/consistency_tests/torch_modules/test_consistency_attention_layer.py
import pytest

from hungarian_net.torch_modules.attention_layer import AttentionLayer

@pytest.mark.consistency
def test_AttentionLayer_init(attentionLayer: AttentionLayer) -> None:
    """Test the initialization of the AttentionLayer.

    Args:
        attentionLayer (AttentionLayer): The AttentionLayer instance provided by the fixture.

    Returns:
        None
    """
    assert isinstance(
        attentionLayer, AttentionLayer
    ), f"AttentionLayer is not an instance of AttentionLayer class, got {attentionLayer.__repr__()}"
