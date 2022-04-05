import pickle
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score 
from GAT import GraphAttentionLayer

class GATNet(torch.nn.Module):
    def __init__(self, model_name, dataset_name, num_features):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        if model_name == 'GAT':
            if dataset_name == "PATTERN":
                self.conv1 = GraphAttentionLayer(num_features, 38, num_heads=8, concat=True, dropout=0.0)
                self.conv2 = GraphAttentionLayer(304,38,num_heads=8,concat=True,dropout=0.0)
                self.conv3 = GraphAttentionLayer(304,2,num_heads=8,concat=False,dropout=0.0)
            elif dataset_name == "Cora":
                self.conv1 = GraphAttentionLayer(num_features, 8, num_heads=8, concat=True, dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,7,num_heads=1,concat=False,dropout=0.6)
            elif dataset_name == "Citeseer":
                self.conv1 = GraphAttentionLayer(num_features, 8, num_heads=8, concat=True, dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,6,num_heads=1,concat=False,dropout=0.6)
            elif dataset_name == "Pubmed":
                self.conv1 = GraphAttentionLayer(num_features,8,num_heads=8,concat=True,dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,3,num_heads=8,concat=False,dropout=0.6)
            elif dataset_name == "AmazonComp":
                self.conv1 = GraphAttentionLayer(num_features,8,num_heads=8,concat=True,dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,10,num_heads=8,concat=False,dropout=0.6)
            elif dataset_name == "AmazonPhotos":
                self.conv1 = GraphAttentionLayer(num_features,8,num_heads=8,concat=True,dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,8,num_heads=8,concat=False,dropout=0.6)
        elif model_name == 'GCN':
            if self.dataset_name == "PATTERN":
                self.conv1 = GCNConv(num_features,304)
                self.conv2 = GCNConv(304,304)
                self.conv3 = GCNConv(304,2)
            elif dataset_name == "Cora":
                self.conv1 = GCNConv(num_features, 64)
                self.conv2 = GCNConv(64,7)
            elif dataset_name == "Citeseer":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,6)
            elif dataset_name == "Pubmed":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,3)
            elif dataset_name == "AmazonComp":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,10)
            elif dataset_name == "AmazonPhotos":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.dataset_name == "PATTERN":
            x = self.conv1(x, edge_index)
            if self.model_name == "GCN":
                x = F.relu(x)
            else:
                x = F.elu(x)
            x = self.conv2(x, edge_index)
            if self.model_name == "GCN":
                x = F.relu(x)
            else:
                x = F.elu(x)
            x = self.conv3(x, edge_index)
            x = F.log_softmax(x, dim=1)
            return x 
        else:
            x = F.dropout(x,p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            if self.model_name == "GCN":
                x = F.relu(x)
            else:
                x = F.elu(x)
            x = F.dropout(x,p=0.6,training=self.training)
            x = self.conv2(x, edge_index)
            x = F.log_softmax(x,dim=1)
            return x 