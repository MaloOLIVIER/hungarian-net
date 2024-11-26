import torch
from train_hnet import HNetGRU, HungarianDataset
from IPython import embed
import matplotlib.pyplot as plot
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import numpy as np


# TODO: understand how the author computed the F1 score


use_cuda = False
max_len = 10

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

model = HNetGRU(max_len=max_len).to(device)
model.eval()

model.load_state_dict(
    torch.load("data/hnet_model.pt", map_location=torch.device("cpu"))
)
test_data = HungarianDataset(train=False, max_len=max_len)

test_f = 0
nb_test_batches = 0
train_dataset = HungarianDataset(train=True, max_len=max_len)
batch_size = 256
f_score_weights = np.tile(train_dataset.get_f_wts(), batch_size)

# load test dataset
test_loader = DataLoader(
    HungarianDataset(train=False, max_len=max_len),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device).float()
        target1 = target[0].to(device).float()
        output1, _, _ = model(data)
        f_pred = (torch.sigmoid(output1).cpu().numpy() > 0.5).reshape(-1)
        f_ref = target1.cpu().numpy().reshape(-1)
        test_f += f1_score(
            f_ref,
            f_pred,
            zero_division=1,
            average="weighted",
            sample_weight=f_score_weights,
        )
        nb_test_batches += 1


# Compute F1 Score
test_f /= nb_test_batches
print(f"F1 Score on Test Data: {test_f:.4f}")

# Plot F1 Score
plot.figure(figsize=(6, 4))
plot.bar(["F1 Score"], [test_f], color="skyblue")
plot.ylim([0, 1])
plot.ylabel("F1 Score")
plot.title("F1 Score for HNet on Test Data")
plot.show()
