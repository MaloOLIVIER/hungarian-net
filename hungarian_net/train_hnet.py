import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from hungarian_net.dataset import HungarianDataset
from hungarian_net.models import HNetGRU


def main(
    batch_size=256,
    nb_epochs=1000,
    max_len=2,
    sample_range_used=[3000, 5000, 15000],
    filename_train="data/reference/hung_data_train",
    filename_test="data/reference/hung_data_test",
):
    """
    Train the Hungarian Network (HNetGRU) model.

    This function orchestrates the training process of the HNetGRU model, including data loading,
    model initialization, training loop with validation, and saving the best-performing model.

    Args:
        batch_size (int, optional):
            Number of samples per training batch. Defaults to 256.
        nb_epochs (int, optional):
            Total number of training epochs. Defaults to 1000.
        max_len (int, optional):
            Maximum number of Directions of Arrival (DOAs) the model can handle. Defaults to 2.
        sample_range_used (List[int], optional):
            List specifying the range of samples used for training. Defaults to [3000, 5000, 15000].
        filename_train (str, optional):
            Path to the training data file. Defaults to "data/reference/hung_data_train".
        filename_test (str, optional):
            Path to the testing data file. Defaults to "data/reference/hung_data_test".

    Steps:
        1. **Set Random Seed**:
            - Ensures reproducibility by setting seeds for Python's `random`, NumPy, and PyTorch.

        2. **Device Configuration**:
            - Checks for GPU availability and sets the computation device accordingly (CUDA or CPU).

        3. **Data Loading**:
            - Loads the training dataset using `HungarianDataset` with specified parameters.
            - Initializes a `DataLoader` for batching and shuffling training data.
            - Calculates class imbalance to handle potential data skew.
            - Loads the validation dataset similarly.

        4. **Model and Optimizer Initialization**:
            - Instantiates the `HNetGRU` model and moves it to the configured device.
            - Sets up the optimizer (`Adam`) for training the model parameters.

        5. **Loss Function Definition**:
            - Defines three separate Binary Cross-Entropy with Logits Loss functions (`criterion1`, `criterion2`, `criterion3`).
            - Assigns equal weights to each loss component.

        6. **Training Loop**:
            - Iterates over the specified number of epochs.
            - **Training Phase**:
                a. Sets the model to training mode.
                b. Iterates over training batches:
                    - Performs forward pass.
                    - Computes individual losses.
                    - Aggregates losses with defined weights.
                    - Backpropagates and updates model weights.
                c. Accumulates and averages training losses.

            - **Validation Phase**:
                a. Sets the model to evaluation mode.
                b. Iterates over validation batches without gradient computation:
                    - Performs forward pass.
                    - Computes losses.
                    - Calculates F1 scores with weighted averaging.
                c. Accumulates and averages validation losses and F1 scores.

            - **Early Stopping**:
                - Monitors validation F1 score to identify and save the best-performing model.
                - Saves model weights with a timestamped filename for version tracking.

            - **Metrics Logging**:
                - Prints comprehensive metrics after each epoch, including losses, F1 scores, and accuracy.
                - Tracks unweighted F1 scores separately for detailed analysis.

        7. **Final Output**:
            - Prints the best epoch and corresponding F1 score.
            - Returns the best-performing model instance.

    Returns:
        HNetGRU:
            The trained HNetGRU model with the best validation F1 score.
    """

    set_seed()

    # Check wether to run on cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # load training dataset
    train_dataset = HungarianDataset(
        train=True, max_len=max_len, filename=filename_train
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    f_score_weights = np.tile(train_dataset.get_f_wts(), batch_size)
    print(train_dataset.get_f_wts())

    # Compute class imbalance
    class_imbalance = train_dataset.compute_class_imbalance()
    print("Class imbalance in training labels:", class_imbalance)

    # load validation dataset
    test_loader = DataLoader(
        HungarianDataset(train=False, max_len=max_len, filename=filename_test),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # load Hnet model and loss functions
    model = HNetGRU(max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters())

    criterion1 = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion2 = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion3 = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion_wts = [1.0, 1.0, 1.0]

    # Start training
    best_f = -1
    best_epoch = -1
    for epoch in range(1, nb_epochs + 1):
        train_start = time.time()
        # TRAINING
        model.train()
        train_loss, train_l1, train_l2, train_l3 = 0, 0, 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device).float()
            target1 = target[0].to(device).float()
            target2 = target[1].to(device).float()
            target3 = target[2].to(device).float()

            optimizer.zero_grad()

            output1, output2, output3 = model(data)

            l1 = criterion1(output1, target1)
            l2 = criterion2(output2, target2)
            l3 = criterion3(output3, target3)
            loss = criterion_wts[0] * l1 + criterion_wts[1] * l2 + criterion_wts[2] * l3

            loss.backward()
            optimizer.step()

            train_l1 += l1.item()
            train_l2 += l2.item()
            train_l3 += l3.item()
            train_loss += loss.item()

        train_l1 /= len(train_loader.dataset)
        train_l2 /= len(train_loader.dataset)
        train_l3 /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        train_time = time.time() - train_start

        # TESTING
        test_start = time.time()
        model.eval()
        test_loss, test_l1, test_l2, test_l3 = 0, 0, 0, 0
        test_f = 0
        nb_test_batches = 0
        true_positives, false_positives, false_negatives = 0, 0, 0
        f1_score_unweighted = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).float()
                target1 = target[0].to(device).float()
                target2 = target[1].to(device).float()
                target3 = target[2].to(device).float()

                output1, output2, output3 = model(data)
                l1 = criterion1(output1, target1)
                l2 = criterion2(output2, target2)
                l3 = criterion3(output3, target3)
                loss = (
                    criterion_wts[0] * l1
                    + criterion_wts[1] * l2
                    + criterion_wts[2] * l3
                )

                test_l1 += l1.item()
                test_l2 += l2.item()
                test_l3 += l3.item()
                test_loss += loss.item()  # sum up batch loss

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

                true_positives += np.sum((f_pred == 1) & (f_ref == 1))
                false_positives += np.sum((f_pred == 1) & (f_ref == 0))
                false_negatives += np.sum((f_pred == 0) & (f_ref == 1))

                f1_score_unweighted += (
                    2
                    * true_positives
                    / (2 * true_positives + false_positives + false_negatives)
                )

        test_l1 /= len(test_loader.dataset)
        test_l2 /= len(test_loader.dataset)
        test_l3 /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        test_f /= nb_test_batches
        test_time = time.time() - test_start
        weighted_accuracy = train_dataset.compute_weighted_accuracy(
            true_positives, false_positives
        )

        f1_score_unweighted /= nb_test_batches

        # Early stopping
        if test_f > best_f:
            best_f = test_f
            best_epoch = epoch

            # Get current date
            current_date = datetime.datetime.now().strftime("%Y%m%d")

            # TODO: change model filename - leverage TensorBoard

            os.makedirs(f"models/{current_date}", exist_ok=True)

            # Human-readable filename
            out_filename = f"models/{current_date}/hnet_model_DOA{max_len}_{'-'.join(map(str, sample_range_used))}.pt"

            torch.save(model.state_dict(), out_filename)

            model_to_return = model
        print(
            "Epoch: {}\t time: {:0.2f}/{:0.2f}\ttrain_loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})\ttest_loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})\tf_scr: {:.4f}\tbest_epoch: {}\tbest_f_scr: {:.4f}\ttrue_positives: {}\tfalse_positives: {}\tweighted_accuracy: {:.4f}".format(
                epoch,
                train_time,
                test_time,
                train_loss,
                train_l1,
                train_l2,
                train_l3,
                test_loss,
                test_l1,
                test_l2,
                test_l3,
                test_f,
                best_epoch,
                best_f,
                true_positives,
                false_positives,
                weighted_accuracy,
            )
        )
        print("F1 Score (unweighted) : {:.4f}".format(f1_score_unweighted))
    print("Best epoch : {}\nBest F1 score : {}".format(best_epoch, best_f))

    return model_to_return


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
