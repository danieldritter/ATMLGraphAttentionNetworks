import pickle
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI, Amazon
from torch_geometric.nn import GATConv, GCNConv, GINConv
from GAT import GraphAttentionLayer
from GATNet import GATNet, EGAT
import os
import networkx as nx
import pandas as pd


torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
# Hyper-Parameters
CUR_DATASET = 'Pubmed'  # Options: Cora, Citeseer, Pubmed, AmazonComp, AmazonPhotos

LEARNING_RATE = 0.01
WEIGHT_DECAY = .0005
# CUR_MODEL = 'GAT'  # Options: GAT, GCN

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 100
NUM_EPOCHS = 5000
LOGGING_FREQUENCY = 10
NUM_RUNS = 20

VERBOSE = True


def adjacent_edges(edges, i, j):
    count = 0
    for x in range(len(edges)):
        for y in range(len(edges)):
            if (x == i or y == i or x == j or y == j) and edges[x][y] == 1:
                count += 1
    return count


# Main code
def main():
    total_avg = 0.0
    total_avg_list = []
    for i in range(NUM_RUNS):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        if VERBOSE:
            print('Starting run number: ' + str(i + 1))
        if CUR_DATASET == "Cora" or CUR_DATASET == "Citeseer" or CUR_DATASET == "Pubmed":
            dataset = Planetoid('./data', CUR_DATASET, split="public", num_train_per_class=20)
            num_features = dataset.num_node_features
            num_classes = dataset.num_classes

        elif CUR_DATASET == 'AmazonComp':
            dataset = Amazon('./data', 'Computers')
            num_features = 767
            num_classes = 10
        elif CUR_DATASET == 'AmazonPhotos':
            dataset = Amazon('./data', 'Photo')
            num_features = 745
            num_classes = 8

        data = dataset[0]
        # Creating an edge matrix of size N * N, entries as 1 if there is an edge and 0 otherwise
        num_nodes = data.x.shape[0]
        edges = torch.zeros(num_nodes, num_nodes)
        edge_features = torch.zeros(data.x.shape[0], data.x.shape[0], 1)
        edge_features = torch.zeros(len(data.edge_index[0]), 1)

        print(torch_geometric.utils.degree(data.edge_index[0]).shape)
        node_deg = torch_geometric.utils.degree(data.edge_index[0])  # degree of all nodes, shape #nodes * 1

        print(f'num nodes: {data.x.shape[0]}')
        print(f'num edges: {len(data.edge_index[0])}')
        for n in range(len(data.edge_index[0])):
            if n % 1000 == 0:
                print(f'processing {n}th edge.')
            i, j = data.edge_index[0][n], data.edge_index[1][n]
            edges[i][j] = 1
            # Create weak topological features for each edge (the # of adjacent edges of the edge)
            count = node_deg[i] + node_deg[j] - 2
            assert count >= 0
            feature = torch.empty(1)
            feature[0] = count
            edge_features[n] = feature

        torch.save(edge_features, './data/'+CUR_DATASET+'_edge_features.pt')
        data.edge_attr = edge_features

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        model = EGAT(CUR_DATASET, num_features, 1).to(device)
        # model = EGAT(num_features).to(device)
        num_edge_features = data.edge_attr[0].shape
        print(num_edge_features)
        if CUR_DATASET == "AmazonComp" or CUR_DATASET == "AmazonPhotos":
            data = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(data).to(device)
        else:
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
            # print(f'pred shape: {pred.shape}, label shape: {data.y.shape}')
            # print(data.edge_index)
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
                        # if VERBOSE:
                        # print('Did not do better at epoch ' + str(epoch + 1) + '.')
                        # print('    Old max: ' + str(cur_max) + '%')
                        # print('    Current score: ' + str(acc) + '%')
                        # print('')
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
                        print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc) + '%')
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
    avg_acc = total_avg / NUM_RUNS
    stddev = np.sqrt(np.var(total_avg_list))
    ci = 1.96 * (stddev / np.sqrt(len(total_avg_list)))
    print('All Results: ' + str(total_avg_list))
    print(f'Total Test Average: {avg_acc} +/- {ci}')


if __name__ == '__main__':
    main()
