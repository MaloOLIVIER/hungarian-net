from generate_hnet_training_data import load_obj
from IPython import embed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import time

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class HNetGRU(nn.Module):
    def __init__(self, max_len=4, hidden_size = 128):
        super().__init__()
        self.nb_gru_layers = 1
        self.gru = nn.GRU(max_len, hidden_size, self.nb_gru_layers, batch_first=True)
        self.attn = AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, max_len)

    def forward(self, query):
        # query - batch x seq x feature

        out, _ = self.gru(query)
        # out - batch x seq x hidden

        out = out.permute((0, 2, 1))
        # out - batch x hidden x seq

        out = self.attn.forward(out)
        # out - batch x hidden x seq

        out = out.permute((0, 2, 1))
        out = torch.tanh(out)
        # out - batch x seq x hidden

        out = self.fc1(out)
        # out - batch x seq x feature

        out1 = out.view(out.shape[0], -1)
        # out1 - batch x (seq x feature)

        out2, _ = torch.max(out, dim=-1)

        out3, _ = torch.max(out, dim=-2)

        # out2 - batch x seq x 1
        return out1.squeeze(), out2.squeeze(), out3.squeeze()


class HungarianDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, max_len=4):
        if train:
            self.data_dict = load_obj('data/hung_data_train')
        else:
            self.data_dict = load_obj('data/hung_data_test')
        self.max_len = max_len

        self.pos_wts = np.ones(self.max_len**2)
        self.f_scr_wts = np.ones(self.max_len**2)
        if train:
            loc_wts = np.zeros(self.max_len**2)
            for i in range(len(self.data_dict)):
                label = self.data_dict[i][3]
                loc_wts += label.reshape(-1)
            self.f_scr_wts = loc_wts / len(self.data_dict)
            self.pos_wts = (len(self.data_dict)-loc_wts) / loc_wts

    def __len__(self):
        return len(self.data_dict)

    def get_pos_wts(self):
        return self.pos_wts

    def get_f_wts(self):
        return self.f_scr_wts

    def compute_class_imbalance(self):
        """
        Computes the class imbalance in the dataset.

        This method iterates over the data dictionary and counts the occurrences
        of each class in the distance assignment matrix (da_mat).

        Returns:
            dict: A dictionary where keys are class labels and values are the 
                  counts of occurrences of each class.
        """
        class_counts = {}
        for key, value in self.data_dict.items():
            nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart = value
            for row in da_mat:
                for elem in row:
                    if elem not in class_counts:
                        class_counts[elem] = 0
                    class_counts[elem] += 1
        return class_counts

    def __getitem__(self, idx):
        feat = self.data_dict[idx][2]
        label = self.data_dict[idx][3]

        label = [label.reshape(-1), label.sum(-1), label.sum(-2)]
        return feat, label
    
    def compute_weighted_accuracy(self, n1star, n0star):
        """
        Compute the weighted accuracy of the model.
        The weighted accuracy is calculated based on the class imbalance in the dataset.
        The weights for each class are determined by the proportion of the opposite class.
        Parameters:
            n1star (int): The number of true positives.
            n0star (int): The number of false positives.
        Returns:
            WA (float): The weighted accuracy of the model.
        
        References:
        Title: How To Train Your Deep Multi-Object Tracker
        Authors: Yihong Xu, Aljosa Osep, Yutong Ban, Radu Horaud, Laura Leal-Taixe, Xavier Alameda-Pineda
        Year: 2020

        URL: https://arxiv.org/abs/1906.06618
        """
        WA = 0
        
        class_counts = self.compute_class_imbalance()
        
        n0 = class_counts.get(0, 0) # number of 0s
        n1 = class_counts.get(1, 0) # number of 1s
        
        w0 = n1/(n0+n1) # weight for class 0
        w1 = 1 - w0 # weight for class 1
        
        WA = (w1*n1star+w0*n0star)/(w1*n1+w0*n0)
        
        return WA

def main():
    batch_size = 256
    nb_epochs = 10
    max_len = 10 # maximum number of events/DOAs you want to the hungarian algo to associate,
    # this is same as 'max_doas' in generate_hnet_training_data.py

    # Check wether to run on cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device:', device)

    # load training dataset
    train_dataset = HungarianDataset(train=True, max_len=max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, drop_last=True)

    f_score_weights = np.tile(train_dataset.get_f_wts(), batch_size)
    print(train_dataset.get_f_wts())

    # Compute class imbalance
    class_imbalance = train_dataset.compute_class_imbalance()
    print("Class imbalance in training labels:", class_imbalance)

    # load validation dataset
    test_loader = DataLoader(
        HungarianDataset(train=False, max_len=max_len),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # load Hnet model and loss functions
    model = HNetGRU(max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters())

    criterion1 = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion2 = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion3 = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion_wts = [1., 1., 1.]

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
            loss = criterion_wts[0]*l1 + criterion_wts[1]*l2 + criterion_wts[2]*l3

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
        train_time = time.time()-train_start


        #TESTING
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
                loss = criterion_wts[0]*l1 + criterion_wts[1]*l2+ criterion_wts[2]*l3

                test_l1 += l1.item()
                test_l2 += l2.item()
                test_l3 += l3.item()
                test_loss += loss.item()  # sum up batch loss

                f_pred = (torch.sigmoid(output1).cpu().numpy() > 0.5).reshape(-1)
                f_ref = target1.cpu().numpy().reshape(-1)
                test_f += f1_score(f_ref, f_pred, zero_division=1, average='weighted', sample_weight=f_score_weights)
                nb_test_batches += 1

                true_positives += np.sum((f_pred == 1) & (f_ref == 1))
                false_positives += np.sum((f_pred == 1) & (f_ref == 0))
                false_negatives += np.sum((f_pred == 0) & (f_ref == 1))
                
                f1_score_unweighted += 2*true_positives/(2*true_positives+false_positives+false_negatives)

        test_l1 /= len(test_loader.dataset)
        test_l2 /= len(test_loader.dataset)
        test_l3 /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        test_f /= nb_test_batches
        test_time = time.time() - test_start
        weighted_accuracy = train_dataset.compute_weighted_accuracy(true_positives, false_positives)
        
        f1_score_unweighted /= nb_test_batches

        # Early stopping
        if test_f > best_loss:
            best_loss = test_f
            best_epoch = epoch
            torch.save(model.state_dict(), "data/hnet_model.pt")
        print('Epoch: {}\t time: {:0.2f}/{:0.2f}\ttrain_loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})\ttest_loss: {:.4f} ({:.4f}, {:.4f}, {:.4f})\tf_scr: {:.4f}\tbest_epoch: {}\tbest_f_scr: {:.4f}\ttrue_positives: {}\tfalse_positives: {}\tweighted_accuracy: {:.4f}'.format(
            epoch, train_time, test_time, train_loss, train_l1, train_l2, train_l3, test_loss, test_l1, test_l2, test_l3, test_f, best_epoch, best_loss, true_positives, false_positives, weighted_accuracy))
        print("F1 Score (unweighted): {:.4f}".format(f1_score_unweighted))
    print('Best epoch: {}\nBest loss: {}'.format(best_epoch, best_loss))


if __name__ == "__main__":
    main()