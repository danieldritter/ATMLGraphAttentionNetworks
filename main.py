
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
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score 
from GAT import GraphAttentionLayer
import argparse 


# Define overall network
class GATNet(torch.nn.Module):
    def __init__(self, model_name, dataset_name, num_features):
        super().__init__()
        self.dataset_name = dataset_name
        if model_name == 'GAT':
            if dataset_name == "PPI":
                self.conv1 = GraphAttentionLayer(num_features, 256, num_heads=4, concat=True, dropout=0.0)
                self.conv2 = GraphAttentionLayer(1024,256,num_heads=4,concat=True,dropout=0.0)
                self.conv3 = GraphAttentionLayer(1024,121,num_heads=6,concat=False,dropout=0.0)
            elif dataset_name == "Cora":
                self.conv1 = GraphAttentionLayer(num_features, 8, num_heads=8, concat=True, dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,7,num_heads=1,concat=False,dropout=0.6)
            elif dataset_name == "Citeseer":
                self.conv1 = GraphAttentionLayer(num_features, 8, num_heads=8, concat=True, dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,6,num_heads=1,concat=False,dropout=0.6)
            elif dataset_name == "Pubmed":
                self.conv1 = GraphAttentionLayer(num_features,8,num_heads=8,concat=True,dropout=0.6)
                self.conv2 = GraphAttentionLayer(64,3,num_heads=8,concat=False,dropout=0.6)
        elif model_name == 'GATGeometric':
            if dataset_name == "PPI":
                self.conv1 = GATConv(num_features, 256, heads=4, concat=True, dropout=0.0)
                self.conv2 = GATConv(1024,256,heads=4,concat=True,dropout=0.0)
                self.conv3 = GATConv(1024,121,heads=6,concat=False,dropout=0.0)
            elif dataset_name == "Cora":
                self.conv1 = GATConv(num_features, 8, heads=8, concat=True, dropout=0.6)
                self.conv2 = GATConv(64,7,heads=1,concat=False,dropout=0.6)
            elif dataset_name == "Citeseer":
                self.conv1 = GATConv(num_features, 8, heads=8, concat=True, dropout=0.6)
                self.conv2 = GATConv(64,6,heads=1,concat=False,dropout=0.6)   
            elif dataset_name == "Pubmed":
                self.conv1 = GATConv(num_features,8,heads=8,concat=True,dropout=0.6)
                self.conv2 = GATConv(64,3,heads=8,concat=False,dropout=0.6)
        elif model_name == 'GCN':
            if self.dataset_name == "PPI":
                print("GCN Benchmark not used for PPI")
                exit()
            elif dataset_name == "Cora":
                self.conv1 = GCNConv(num_features, 64)
                self.conv2 = GCNConv(64,7)
            elif dataset_name == "Citeseer":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,6)
            elif dataset_name == "Pubmed":
                self.conv1 = GCNConv(num_features,64)
                self.conv2 = GCNConv(64,3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.dataset_name == "PPI":
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = self.conv3(x, edge_index)
            x = torch.sigmoid(x)
            return x 
        else:
            x = F.dropout(x,p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x,p=0.6,training=self.training)
            x = self.conv2(x, edge_index)
            x = F.log_softmax(x,dim=1)
            return x 


# Main code
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--model", default="GAT")
    args = parser.parse_args()
    use_early_stopping = True 
    early_stopping_patience = 100 
    num_runs = 5
    num_epochs = 10000
    verbose = True 
    logging_frequency = 10
    num_forced_epochs = 20 
    if args.dataset == "PPI":
        lr = .005 
        l2_lambda = 0.0 
        dropout = 0.0 
        batch_size = 1 
    elif args.dataset == "Pubmed":
        lr = .01
        l2_lambda = .001 
        dropout = 0.6
        batch_size = 1
    else:
        lr = .005 
        l2_lambda = .0005 
        dropout = 0.6 
        batch_size = 1 

    total_avg = 0.0
    total_avg_list = []
    for i in range(num_runs):
        val_losses = [] 
        val_accs = []
        if verbose:
            print('Starting run number: ' + str(i + 1))
        
        use_planetoid = args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'

        if use_planetoid:
            dataset = Planetoid('./data', args.dataset, split="public", num_train_per_class=20, transform=T.NormalizeFeatures())
            num_features = dataset.num_node_features
            num_classes = dataset.num_classes
        elif args.dataset == 'AmazonComp':
            dataset = Amazon('./data', 'Computers', transform=T.NormalizeFeatures())
            num_features = 767
            num_classes = 10
        elif args.dataset == 'AmazonPhotos':
            dataset = Amazon('./data', 'Photo', transform=T.NormalizeFeatures())
            num_features = 745
            num_classes = 8
        elif args.dataset == 'PPI':
            datasetTrain = PPI('./data', 'train')
            datasetVal = PPI('./data', 'val')
            datasetTest = PPI('./data', 'test')
            num_features = 50
            num_classes = 121
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        model = GATNet(args.model, args.dataset, num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
        if verbose:
            print('Starting training...')
        if args.dataset == "PPI":
            train_loader = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(datasetVal, batch_size=batch_size)
            test_loader = DataLoader(datasetTest, batch_size=batch_size)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size)
            val_loader = train_loader 
            test_loader = train_loader 
        epoch = 0
        stop_counter = 0
        cur_max = 0.0
        cur_min_loss = float("inf")
        stop_training = False
        while not stop_training:
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                if args.dataset == 'PPI':
                    loss = F.binary_cross_entropy(out, batch.y)
                else:
                    loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
            model.eval()                
            if use_early_stopping:
                if epoch >= num_forced_epochs - 1:
                    losses = 0.0
                    accs = 0.0
                    preds = []
                    labels = []
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch)
                        if args.dataset == 'PPI':
                            pred = (out>0.5).int()
                            preds.append(pred)
                            labels.append(batch.y)
                            loss = F.binary_cross_entropy(out, batch.y)
                            losses += loss.item()
                        else:
                            pred = out.argmax(dim=1)
                            loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
                            losses += loss.item()
                            correct = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum()
                            acc = (correct / batch.val_mask.sum()).item()
                            accs += acc
                    if args.dataset == "PPI":
                        avg_acc = f1_score(torch.cat(labels,dim=0).flatten(), torch.cat(preds,dim=0).flatten(), average="micro")
                    else:
                        avg_acc = accs/len(val_loader)
                    avg_loss = losses/len(val_loader)
                    val_losses.append(avg_loss)
                    val_accs.append(avg_acc)
                    if avg_acc >= cur_max or avg_loss <= cur_min_loss:
                        if verbose:
                            print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                            print('    Old max acc: ' + str(cur_max) + '%')
                            print('    New max acc: ' + str(avg_acc) + '%')
                            print('    Old max loss: ' + str(cur_min_loss) + '%')
                            print('    New max loss: ' + str(avg_loss) + '%')
                            print('')
                        if avg_acc >= cur_max and avg_loss <= cur_min_loss:
                            torch.save(model.state_dict(), "./model/cur_model.pt")
                        cur_max = max(avg_acc, cur_max)
                        cur_min_loss = min(cur_min_loss, avg_loss)
                        stop_counter = 0
                    else:
                        stop_counter = stop_counter + 1
                        if verbose:
                            print('Did not do better at epoch ' + str(epoch + 1) + '.')
                            print('    Old max: ' + str(cur_max) + '%')
                            print('    Current score: ' + str(avg_acc) + '%')
                            print('')
                        if stop_counter >= early_stopping_patience:
                            if verbose:
                                print('Stopping training...')
                            stop_training = True
            else:
                if verbose:
                    if not epoch == 0 and (epoch + 1) % logging_frequency == 0:
                        losses = 0.0 
                        accs = 0.0 
                        for batch in val_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            if args.dataset == 'PPI':
                                loss = F.binary_cross_entropy(out, batch.y)
                                losses += loss.item()
                                pred = (out>0.5).int()
                                acc = f1_score(batch.y.cpu().detach(), pred.cpu().detach(), average="micro")
                                accs += acc
                            else:
                                pred = out.argmax(dim=1)
                                loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
                                losses += loss.item()
                                correct = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum()
                                acc = float(int(correct) / int(batch.val_mask.sum()))
                                accs += acc 
                        val_accs.append(accs/len(val_loader))
                        val_losses.append(losses/len(val_loader))
                        print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc) + '%')
                if epoch >= num_epochs - 1:
                    stop_training = True
            epoch = epoch + 1
        gen_graph(val_accs, "validation_accuracy", i)
        gen_graph(val_losses, "validation_losses", i)
        model.eval()
        if use_early_stopping:
            model.load_state_dict(torch.load("./model/cur_model.pt"))
        if args.dataset == 'PPI':
            accs = 0.0 
            for batch in test_loader:
                batch = batch.to(device)
                pred = (model(batch)>0.5).int()
                acc = f1_score(batch.y.cpu().detach(), pred.cpu().detach(), average="micro")
                accs += acc
        else:
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=1)
                correct = (pred[batch.test_mask] == batch.y[batch.test_mask]).sum()
                acc = float(int(correct) / int(batch.test_mask.sum()))

        print(f'Test Accuracy: {accs/len(test_loader):.4f}%')
        total_avg += accs/len(test_loader)
        total_avg_list.append(accs/len(test_loader))

    print('All Results: ' + str(total_avg_list))
    print(f'Total Test Average: {total_avg/num_runs}')

if __name__ == '__main__':
    main()
