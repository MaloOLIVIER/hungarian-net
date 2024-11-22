import torch
from train_hnet import HNetGRU, AttentionLayer, load_best_f_score

import torch.nn as nn

def test_forward():
    # Initialize the model
    max_len = 4
    model = HNetGRU(max_len=max_len)

    # Create a dummy input tensor with batch size 2, sequence length 4, and feature size 4
    input_tensor = torch.randn(2, max_len, max_len)
    
    # print("input_tensor", input_tensor)

    # Forward pass
    out1, out2, out3 = model.forward(input_tensor)
    
    # print("out1", out1)
    # print("out2", out2)
    # print("out3", out3)

    # Check the shapes of the outputs
    assert out1.shape == (2, max_len * max_len), f"Expected shape (2, {max_len * max_len}), but got {out1.shape}"
    assert out2.shape == (2, max_len), f"Expected shape (2, {max_len}), but got {out2.shape}"
    assert out3.shape == (2, max_len), f"Expected shape (2,), but got {out3.shape}"

    print("Forward test passed!")

    import torch.nn as nn

def test_attention_layer():
    # Initialize the AttentionLayer
    in_channels = 4
    out_channels = 4
    key_channels = 4
    attention_layer = AttentionLayer(in_channels, out_channels, key_channels)

    # Create a dummy input tensor with batch size 2, in_channels 4, and sequence length 4
    input_tensor = torch.randn(2, in_channels, 4)

    # Forward pass
    output = attention_layer.forward(input_tensor)

    # Check the shape of the output
    assert output.shape == (2, out_channels, 4), f"Expected shape (2, {out_channels}, 4), but got {output.shape}"

    print("Attention layer test passed!")

import unittest
from train_hnet import HungarianDataset, DataLoader, f1_score
import numpy as np
import time

class TestHNetModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 10
        self.batch_size = 256
        self.model = HNetGRU(max_len=self.max_len).to(self.device)
        self.model.load_state_dict(torch.load("data/hnet_model.pt", map_location=torch.device(self.device)))
        self.test_dataset = HungarianDataset(train=False, max_len=self.max_len)
        self.train_dataset = HungarianDataset(train=True, max_len=self.max_len)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.f_score_weights = np.tile(self.train_dataset.get_f_wts(), self.batch_size)

    def test_f_score(self):
        last_best_f_score = load_best_f_score()
        self.model.eval()
        test_f = 0
        nb_test_batches = 0
        start = time.time()
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device).float()
                target1 = target[0].to(self.device).float()

                output1, _, _ = self.model(data)
                f_pred = (torch.sigmoid(output1).cpu().numpy() > 0.5).reshape(-1)
                f_ref = target1.cpu().numpy().reshape(-1)
                test_f += f1_score(f_ref, f_pred, zero_division=1, average='weighted', sample_weight=self.f_score_weights)
                nb_test_batches += 1

        test_f /= nb_test_batches
        end = time.time()
        self.assertGreater(test_f, last_best_f_score, "F1 Score is lower than expected, possible regression detected.")
        print("F1 Score on Test Data: ", test_f)
        print("Theoretical time taken for inference : {:.2f} ms".format((end - start) / nb_test_batches * 1000))

if __name__ == "__main__":
    print('\n')
    test_forward()
    test_attention_layer()
    unittest.main()