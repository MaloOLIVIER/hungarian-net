from hungarian_net.generate_hnet_training_data import load_obj
from IPython import embed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
import time

from hungarian_net.models import AttentionLayer, HNetGRU
from hungarian_net.dataset import HungarianDataset


def main():
    filename_train = "train_data_dict.pkl"
    filename_test = "test_data_dict.pkl"
    batch_size = 256
    nb_epochs = 1000
    max_len = (
        2  # maximum number of events/DOAs you want the hungarian algo to associate,
    )

    # Check wether to run on cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # load training dataset
    train_dataset = HungarianDataset(train=True, max_len=max_len, filename=filename)
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
        HungarianDataset(train=False, max_len=max_len),
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
    best_loss = -1
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
        if test_f > best_loss:
            best_loss = test_f
            best_epoch = epoch
            torch.save(model.state_dict(), "data/hnet_model.pt")
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
                best_loss,
                true_positives,
                false_positives,
                weighted_accuracy,
            )
        )
        print("F1 Score (unweighted): {:.4f}".format(f1_score_unweighted))
    print("Best epoch: {}\nBest loss: {}".format(best_epoch, best_loss))


if __name__ == "__main__":
    main()
