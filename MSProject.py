
# Library imports
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import GATConv

# Hyper-Parameters
USE_PLANETOID = False
PLANETOID_SET = 'Pubmed'
USE_PPI = True
LEARNING_RATE = 0.01
HIDDEN_FEATURES = 64

# Define overall network
class GATNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, HIDDEN_FEATURES)
        self.conv2 = GATConv(HIDDEN_FEATURES, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        if USE_PPI:
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)

        return x

# Main code
def main():

    if USE_PLANETOID:
        dataset = Planetoid('./data', PLANETOID_SET)
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes
    if USE_PPI:
        datasetTrain = PPI('./data', 'train')
        datasetTest = PPI('./data', 'test')
        num_features = 50
        num_classes = 121

    device = torch.device('cpu')
    model = GATNet(num_features, num_classes).to(device)
    if USE_PPI:
        data = datasetTrain[0].to(device)
    else:
        data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        if USE_PPI:
            loss = F.cross_entropy(out, data.y)
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    if USE_PPI:
        data  = datasetTest[0].to(device)
        model.eval()
        pred = (model(data)>0.5).int()
        print(pred[43])
        print(data.y[43])
        correct = (pred == data.y).sum()
        acc = int(correct) / int(data.y.shape[0] * 121)
    else:
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()

