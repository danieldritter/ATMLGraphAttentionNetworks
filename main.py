
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
    def __init__(self, num_features, num_hidden_features, num_layers, num_classes, num_heads, concats, model_name, dataset_name, dropout=0.6):
        super().__init__()
        self.dropout_val = dropout 
        self.conv_layers = torch.nn.ModuleList()
        self.dataset_name = dataset_name
        self.num_layers = num_layers
        if model_name == "GCN":
            concats = [False for i in range(num_layers)]
        if model_name == 'GAT':
            self.conv_layers.append(GraphAttentionLayer(num_features, num_hidden_features, num_heads[0], concat=concats[0], dropout=dropout))
            for i in range(1,num_layers-1):
                if concats[i-1]:
                    self.conv_layers.append(GraphAttentionLayer(num_hidden_features*num_heads[i-1], num_hidden_features, num_heads[i], concat=concats[i], dropout=dropout))
                else:
                    self.conv_layers.append(GraphAttentionLayer(num_hidden_features, num_hidden_features, num_heads=num_heads[i], concat=concats[i], dropout=dropout))
            if concats[-2]:
                self.conv_layers.append(GraphAttentionLayer(num_hidden_features*num_heads[-2], num_classes, num_heads[-1], concat=concats[-1], dropout=dropout))
            else:
                self.conv_layers.append(GraphAttentionLayer(num_hidden_features, num_classes, num_heads[-1], concat=concats[-1], dropout=dropout))
        elif model_name == 'GATGeometric':
            self.conv_layers.append(GATConv(num_features, num_hidden_features, num_heads[0], concat=concats[0], dropout=dropout))
            for i in range(1,num_layers-1):
                if concats[i-1]:
                    self.conv_layers.append(GATConv(num_hidden_features*num_heads[i-1], num_hidden_features, num_heads[i], concat=concats[i], dropout=dropout))
                else:
                    self.conv_layers.append(GATConv(num_hidden_features, num_hidden_features, heads=num_heads[i], concat=concats[i], dropout=dropout))
            if concats[-2]:
                self.conv_layers.append(GATConv(num_hidden_features*num_heads[-2], num_classes, heads=num_heads[-1], concat=concats[-1], dropout=dropout))
            else:
                self.conv_layers.append(GATConv(num_hidden_features, num_classes, heads=num_heads[-1], concat=concats[-1], dropout=dropout))
        elif model_name == 'GCN':
            self.conv_layers.append(GCNConv(num_features, num_hidden_features*num_heads[0], dropout=dropout))
            for i in range(num_layers-2):
                if concats[i-1]:
                    self.conv_layers.append(GCNConv(num_hidden_features*num_heads[i-1], num_hidden_features*num_heads[i], dropout=dropout))
                else:
                    self.conv_layers.append(GCNConv(num_hidden_features, num_hidden_features*num_heads[i], dropout=dropout))
            if concats[-2]:
                self.conv_layers.append(GCNConv(num_hidden_features*num_heads[-2], num_classes, dropout=dropout))
            else:
                self.conv_layers.append(GCNConv(num_hidden_features, num_classes, dropout=dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout_val, training=self.training)
            x = self.conv_layers[i](x, edge_index)
            x = F.elu(x)
        if self.dataset_name == 'PPI':
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)

        return x


# Main code
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--model", default="GAT")
    parser.add_argument("--learning_rate", type=float, default=.005)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_hidden_features", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, nargs="+")
    parser.add_argument("--concat_layers", type=int, nargs="+")
    parser.add_argument("--l2_lambda", type=float, default=.0005)
    parser.add_argument("--dropout_val", type=float, default=0.6)
    parser.add_argument("--use_early_stopping", action='store_true')
    parser.add_argument("--early_stopping_patience", default=100)
    parser.add_argument("--num_forced_epochs", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--logging_frequency", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    args.concat_layers = [True if x == 1 else False for x in args.concat_layers]
    assert len(args.attention_heads) == args.num_layers == len(args.concat_layers)
    total_avg = 0.0
    total_avg_list = []
    for i in range(args.num_runs):
        val_losses = [] 
        val_accs = []
        if args.verbose:
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

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        model = GATNet(num_features, args.num_hidden_features, args.num_layers, num_classes, args.attention_heads, args.concat_layers, args.model, args.dataset, args.dropout_val).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)
        if args.verbose:
            print('Starting training...')
        if args.dataset == "PPI":
            train_loader = DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(datasetVal, batch_size=args.batch_size)
            test_loader = DataLoader(datasetTest, batch_size=args.batch_size)
        else:
            train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
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
            if args.use_early_stopping:
                if epoch >= args.num_forced_epochs - 1:
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
                        if args.verbose:
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
                        if args.verbose:
                            print('Did not do better at epoch ' + str(epoch + 1) + '.')
                            print('    Old max: ' + str(cur_max) + '%')
                            print('    Current score: ' + str(avg_acc) + '%')
                            print('')
                        if stop_counter >= args.early_stopping_patience:
                            if args.verbose:
                                print('Stopping training...')
                            stop_training = True
            else:
                if args.verbose:
                    if not epoch == 0 and (epoch + 1) % args.logging_frequency == 0:
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
                if epoch >= args.num_epochs - 1:
                    stop_training = True
            epoch = epoch + 1
        gen_graph(val_accs, "validation_accuracy", i)
        gen_graph(val_losses, "validation_losses", i)
        model.eval()
        if args.use_early_stopping:
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
    print(f'Total Test Average: {total_avg/args.num_runs}')

if __name__ == '__main__':
    main()
