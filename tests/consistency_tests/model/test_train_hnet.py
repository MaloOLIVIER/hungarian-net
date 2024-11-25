# tests/model/test_train_hnet.py
import pytest
import torch
import numpy as np
from hungarian_net.train_hnet import HNetGRU, AttentionLayer

#TODO: finish to test code robustess

def test_model_initialization(model, max_doas):
    """_summary_

    Args:
        model (_type_): _description_
        max_doas (_type_): _description_
    """
    assert isinstance(model, HNetGRU), "Model is not an instance of HNetGRU"
    assert model.max_len == max_doas, f"Expected max_doas {max_doas}, got {model.max_len}"

# def test_forward_pass(model):
#     input_tensor = torch.randn(1, model.max_len, 10)
#     output = model.forward(input_tensor)
#     assert isinstance(output, tuple), "Output should be a tuple"
#     assert len(output) == 3, f"Expected 3 outputs, got {len(output)}"

# def test_training_step(model, batch_size, sample_data):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.CrossEntropyLoss()
    
#     inputs = torch.tensor(sample_data['features'], dtype=torch.float32)
#     labels = torch.tensor(sample_data['labels'], dtype=torch.long)
    
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
    
#     assert loss.item() >= 0, "Loss should be non-negative"

# def test_model_convergence(model, sample_data):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
    
#     inputs = torch.tensor(sample_data['features'], dtype=torch.float32)
#     labels = torch.tensor(sample_data['labels'], dtype=torch.long)
    
#     for epoch in range(10):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
    
#     # Assuming loss should decrease below a threshold
#     assert loss.item() < 0.5, f"Model did not converge, final loss: {loss.item()}"