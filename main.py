
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
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import GATConv, GCNConv, GINConv


# Hyper-Parameters
CUR_DATASET = 'PPI' # Options: Cora, Citeseer, Pubmed, PPI

LEARNING_RATE = 0.01
HIDDEN_FEATURES = 64
CUR_MODEL = 'GAT' # Options: GAT, GCN, GIN

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 20
STOPPING_CRITERIA = 15
NUM_EPOCHS = 100
LOGGING_FREQUENCY = 10

VERBOSE = True

# Helper variables
use_planetoid = CUR_DATASET == 'Cora' or CUR_DATASET == 'Citeseer' or CUR_DATASET == 'Pubmed'


# Define overall network
class GATNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        global CUR_MODEL
        if CUR_MODEL == 'GAT':
          self.conv1 = GATConv(num_features, HIDDEN_FEATURES)
          self.conv2 = GATConv(HIDDEN_FEATURES, num_classes)
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
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        if CUR_DATASET == 'PPI':
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)

        return x


# Main code
def main():
    global use_planetoid
    if use_planetoid:
        dataset = Planetoid('./data', CUR_DATASET)
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes
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
        data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    if VERBOSE:
        print('Starting training...')

    model.train()
    epoch = 0
    stop_counter = 0
    cur_max = 0.0
    stop_training = False
    while not stop_training:
        optimizer.zero_grad()
        out = model(data)
        if CUR_DATASET == 'PPI':
            loss = F.cross_entropy(out, data.y)
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if USE_EARLY_STOPPING:
            if epoch >= FORCED_EPOCHS - 1:
                model.eval()
                if CUR_DATASET == 'PPI':
                    pred = (model(dataVal)>0.5).int()
                    correct = (pred == dataVal.y).sum()
                    acc = float(int(correct) / int(dataVal.y.shape[0] * num_classes))
                else:
                    pred = model(data).argmax(dim=1)
                    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                    acc = float(int(correct) / int(data.val_mask.sum()))
                if acc >= cur_max:
                    if VERBOSE:
                        print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                        print('    Old max: ' + str(cur_max) + '%')
                        print('    New max: ' + str(acc) + '%')
                        print('')
                    cur_max = acc
                    stop_counter = 0
                    file_h = open('./model/cur_model.p', 'wb')
                    pickle.dump(model, file_h)
                    file_h.close()
                else:
                    stop_counter = stop_counter + 1
                    if VERBOSE:
                        print('Did not do better at epoch ' + str(epoch + 1) + '.')
                        print('    Old max: ' + str(cur_max) + '%')
                        print('    Current score: ' + str(acc) + '%')
                        print('')
                    if stop_counter >= STOPPING_CRITERIA:
                        file_h = open('./model/cur_model.p', 'rb')
                        model = pickle.load(file_h)
                        file_h.close()
                        if VERBOSE:
                            print('Stopping training...')
                        stop_training = True
        else:
            if VERBOSE:
                if not epoch == 0 and (epoch + 1) % LOGGING_FREQUENCY == 0:
                    model.eval()
                    if CUR_DATASET == 'PPI':
                        pred = (model(dataVal)>0.5).int()
                        correct = (pred == dataVal.y).sum()
                        acc = float(int(correct) / int(dataVal.y.shape[0] * num_classes))
                    else:
                        pred = model(data).argmax(dim=1)
                        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                        acc = float(int(correct) / int(data.val_mask.sum()))
                    print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc) + '%')
            if epoch >= NUM_EPOCHS - 1:
                stop_training = True
        epoch = epoch + 1

    model.eval()
    if CUR_DATASET == 'PPI':
        pred = (model(dataTest)>0.5).int()
        correct = (pred == dataTest.y).sum()
        acc = float(int(correct) / int(dataTest.y.shape[0] * num_classes))
    else:
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = float(int(correct) / int(data.test_mask.sum()))

    print(f'Test Accuracy: {acc:.4f}%')

if __name__ == '__main__':
    main()

