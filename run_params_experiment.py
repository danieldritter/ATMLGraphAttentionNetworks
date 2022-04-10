import pickle
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from GAT import GraphAttentionLayer

"""
Runs an experiment testing the change in performance as the number of hidden features in the GAT layer
increases.
"""

class GATModel(torch.nn.Module):

    def __init__(self, num_input_features, num_output_features, num_heads, num_classes):
        super().__init__()
        self.conv1 = GraphAttentionLayer(num_input_features, num_output_features, num_heads=num_heads, concat=True)
        self.conv2 = GraphAttentionLayer(num_output_features*num_heads, num_classes, num_heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


# Hyper-Parameters
CUR_DATASET = 'Cora' # Options: Cora, Citeseer
LEARNING_RATE = 0.005
WEIGHT_DECAY = .0005
CUR_MODEL = 'GAT' # Options: GAT

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 100
NUM_EPOCHS = 10000
LOGGING_FREQUENCY = 10
NUM_RUNS = 20

VERBOSE = True


# Main code
def main():
    heads_feat_pairs = [(2,8),(4,8),(8,8),(16,8),(32,8)]
    all_avgs = {}
    for pair in heads_feat_pairs:
        num_heads = pair[0]
        num_feats = pair[1]
        total_avg = 0.0
        total_avg_list = []
        for i in range(NUM_RUNS):
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            if VERBOSE:
                print('Starting run number: ' + str(i + 1))
            dataset = Planetoid('./data', CUR_DATASET, split="public", num_train_per_class=20)
            num_features = dataset.num_node_features
            num_classes = dataset.num_classes
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = GATModel(num_features,num_feats,num_heads,num_classes).to(device)
            data = dataset[0]
            data = T.NormalizeFeatures()(data).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            if VERBOSE:
                print('Starting training...')

            epoch = 0
            stop_counter = 0
            cur_max = 0.0
            cur_min_loss = float("inf")
            stop_training = False
            while not stop_training:
                model.train()
                optimizer.zero_grad()
                out = model(data)
                pred = out.argmax(dim=1)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
                acc = (correct / data.train_mask.sum()).item()
                train_losses.append(loss.item())
                train_accs.append(acc)
                loss.backward()
                optimizer.step()
                if USE_EARLY_STOPPING:
                    if epoch >= FORCED_EPOCHS - 1:
                        model.eval()
                        out = model(data)
                        pred = out.argmax(dim=1)
                        # print(f'pred shape: {pred.shape}, label shape: {data.y.shape}')
                        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                        acc = (correct / data.val_mask.sum()).item()
                        val_losses.append(loss.item())
                        val_accs.append(acc)
                        if acc >= cur_max or loss.item() <= cur_min_loss:
                            if VERBOSE:
                                print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                                print('    Old max acc: ' + str(cur_max) + '%')
                                print('    New max acc: ' + str(acc) + '%')
                                print('    Old min loss: ' + str(cur_min_loss) + '%')
                                print('    New min loss: ' + str(loss.item()) + '%')
                                print('')
                            if acc > cur_max and loss.item() < cur_min_loss:
                                torch.save(model.state_dict(), "./model/cur_model.pt")
                            cur_max = max(acc, cur_max)
                            cur_min_loss = min(cur_min_loss, loss.item())
                            stop_counter = 0
                        else:
                            stop_counter = stop_counter + 1
                            if stop_counter >= EARLY_STOPPING_PATIENCE:
                                if VERBOSE:
                                    print('Stopping training...')
                                stop_training = True
                else:
                    if VERBOSE:
                        if not epoch == 0 and (epoch + 1) % LOGGING_FREQUENCY == 0:
                            model.eval()
                            out = model(data)
                            pred = out.argmax(dim=1)
                            loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                            acc = float(int(correct) / int(data.val_mask.sum()))
                            val_accs.append(acc)
                            val_losses.append(loss.item())
                            print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc.item()) + '%')
                    if epoch >= NUM_EPOCHS - 1:
                        stop_training = True
                epoch = epoch + 1
            model.eval()
            if USE_EARLY_STOPPING:
                model.load_state_dict(torch.load("./model/cur_model.pt"))
            pred = model(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = float(int(correct) / int(data.test_mask.sum()))

            print(f'Test Accuracy: {acc:.4f}%')
            total_avg += acc
            total_avg_list.append(acc)
        avg_acc = total_avg/NUM_RUNS
        stddev = np.sqrt(np.var(total_avg_list))
        ci = 1.96*(stddev/np.sqrt(len(total_avg_list)))
        print('All Results: ' + str(total_avg_list))
        print(f'Total Test Average: {avg_acc} +/- {ci}')
        all_avgs[pair] = f"{avg_acc} +/- {ci}"
    for pair in all_avgs:
        print(f"Num Features: {pair[0]*pair[1]}, Test Accuracy: {all_avgs[pair]}")

if __name__ == '__main__':
    main()
