
##################################################################
#                                                                #
#            ATML Project - Graph Attention Networks             #
#                                                                #
# Ouns El Harzli, Yangyuqing Li, Daniel Ritter, Dylan Sandfelder #
#                                                                #
##################################################################

# Library imports
import pickle
import torch
from utils import gen_graph
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI, Amazon
from torch_geometric.nn import GATConv, GCNConv, GINConv
from GAT import GraphAttentionLayer


# Hyper-Parameters
CUR_DATASET = 'Cora' # Options: Cora, Citeseer, Pubmed, PPI, AmazonComp, AmazonPhotos

LEARNING_RATE = 0.005
WEIGHT_DECAY = .0005
HIDDEN_FEATURES = 8
CUR_MODEL = 'GAT' # Options: GAT, GATGeometric, GCN, GIN

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 20
STOPPING_CRITERIA = 100
NUM_EPOCHS = 100
LOGGING_FREQUENCY = 10
NUM_RUNS = 10

VERBOSE = True

# Helper variables
use_planetoid = CUR_DATASET == 'Cora' or CUR_DATASET == 'Citeseer' or CUR_DATASET == 'Pubmed'


# Define overall network
class GATNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        global CUR_MODEL
        if CUR_MODEL == 'GAT':
            self.conv1 = GraphAttentionLayer(num_features, HIDDEN_FEATURES, num_heads=8, concat=True)
            self.conv2 = GraphAttentionLayer(HIDDEN_FEATURES*8, num_classes, num_heads=1)
        elif CUR_MODEL == 'GATGeometric':
            self.conv1 = GATConv(num_features, HIDDEN_FEATURES, heads=8, dropout=0.6)
            self.conv2 = GATConv(HIDDEN_FEATURES*8, num_classes, heads=1,  concat=False, dropout=0.6)
        elif CUR_MODEL == 'GCN':
            self.conv1 = GCNConv(num_features, HIDDEN_FEATURES)
            self.conv2 = GCNConv(HIDDEN_FEATURES, num_classes)
        elif CUR_MODEL == 'GIN':
            self.conv1 = GINConv(num_features, HIDDEN_FEATURES)
            self.conv2 = GINConv(HIDDEN_FEATURES, num_classes)

    def forward(self, data):
        global CUR_DATASET

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        if CUR_DATASET == 'PPI':
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)

        return x


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

        global use_planetoid
        if use_planetoid:
            dataset = Planetoid('./data', CUR_DATASET)
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
        elif CUR_DATASET == 'PPI':
            datasetTrain = PPI('./data', 'train')
            datasetVal = PPI('./data', 'val')
            datasetTest = PPI('./data', 'test')
            num_features = 50
            num_classes = 121

        device = torch.device('cpu')
        model = GATNet(num_features, num_classes).to(device)
        if CUR_DATASET == 'PPI':
            data = datasetTrain[0].to(device)
            dataVal = datasetVal[0].to(device)
            dataTest = datasetTest[0].to(device)
        else:
            data = dataset[0]
            data = T.RandomNodeSplit(split='random', num_train_per_class=20, num_val=500, num_test=1000)(data).to(device)
        data = T.NormalizeFeatures()(data)
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
            if CUR_DATASET == 'PPI':
                pred = (out>0.5).int()
                loss = F.cross_entropy(out, data.y)
                correct = (pred == dataVal.y).sum()
                acc = float(int(correct) / int(data.y.shape[0] * num_classes))
            else:
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
                    if CUR_DATASET == 'PPI':
                        out = model(dataVal)
                        pred = (out>0.5).int()
                        loss = F.cross_entropy(out, dataVal.y)
                        correct = (pred == dataVal.y).sum()
                        acc = float(int(correct) / int(dataVal.y.shape[0] * num_classes))
                    else:
                        out = model(data)
                        pred = out.argmax(dim=1)
                        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                        acc = (correct / data.val_mask.sum()).item()
                    val_losses.append(loss.item())
                    val_accs.append(acc)
                    if acc > cur_max or loss.item() < cur_min_loss:
                        if VERBOSE:
                            print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                            print('    Old max acc: ' + str(cur_max) + '%')
                            print('    New max acc: ' + str(acc) + '%')
                            print('    Old max loss: ' + str(cur_min_loss) + '%')
                            print('    New max acc: ' + str(loss.item()) + '%')
                            print('')
                        if acc > cur_max and loss.item() < cur_min_loss:
                            torch.save(model.state_dict(), "./model/cur_model.pt")
                        cur_max = max(acc, cur_max)
                        cur_min_loss = min(cur_min_loss, loss.item())
                        stop_counter = 0
                    else:
                        stop_counter = stop_counter + 1
                        if VERBOSE:
                            print('Did not do better at epoch ' + str(epoch + 1) + '.')
                            print('    Old max: ' + str(cur_max) + '%')
                            print('    Current score: ' + str(acc) + '%')
                            print('')
                        if stop_counter >= STOPPING_CRITERIA:
                            if VERBOSE:
                                print('Stopping training...')
                            stop_training = True
            else:
                if VERBOSE:
                    if not epoch == 0 and (epoch + 1) % LOGGING_FREQUENCY == 0:
                        model.eval()
                        if CUR_DATASET == 'PPI':
                            out = model(dataVal)
                            loss = F.cross_entropy(out, dataVal.y)
                            pred = (out>0.5).int()
                            correct = (pred == dataVal.y).sum()
                            acc = float(int(correct) / int(dataVal.y.shape[0] * num_classes))
                        else:
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
        gen_graph(train_accs, "train_accuracy", i)
        gen_graph(train_losses, "train_losses", i)
        gen_graph(val_accs, "validation_accuracy", i)
        gen_graph(val_losses, "validation_losses", i)
        model.eval()
        if USE_EARLY_STOPPING:
            model.load_state_dict(torch.load("./model/cur_model.pt"))
        if CUR_DATASET == 'PPI':
            pred = (model(dataTest)>0.5).int()
            correct = (pred == dataTest.y).sum()
            acc = float(int(correct) / int(dataTest.y.shape[0] * num_classes))
        else:
            pred = model(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = float(int(correct) / int(data.test_mask.sum()))

        print(f'Test Accuracy: {acc:.4f}%')
        total_avg += acc
        total_avg_list.append(acc)

    print('All Results: ' + str(total_avg_list))
    print(f'Total Test Average: {total_avg/NUM_RUNS}')

if __name__ == '__main__':
    main()
