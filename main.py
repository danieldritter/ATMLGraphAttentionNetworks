
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
            for i in range(num_layers-2):
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
        # elif model_name == 'GIN':
        #     self.conv1 = GINConv(num_features, HIDDEN_FEATURES)
        #     self.conv2 = GINConv(HIDDEN_FEATURES, num_classes)

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
    parser.add_argument("--num_hidden_features", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, nargs="+")
    parser.add_argument("--concat_layers", type=int, nargs="+")
    parser.add_argument("--l2_lambda", type=float, default=.0005)
    parser.add_argument("--dropout_val", type=float, default=0.6)
    parser.add_argument("--use_early_stopping", action='store_true')
    parser.add_argument("--early_stopping_patience", default=100)
    parser.add_argument("--num_forced_epochs", type=int, default=20)
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
        train_losses = [] 
        train_accs = [] 
        val_losses = [] 
        val_accs = []
        if args.verbose:
            print('Starting run number: ' + str(i + 1))
        
        use_planetoid = args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'

        if use_planetoid:
            dataset = Planetoid('./data', args.dataset, split="public", num_train_per_class=20)
            num_features = dataset.num_node_features
            num_classes = dataset.num_classes
        elif args.dataset == 'AmazonComp':
            dataset = Amazon('./data', 'Computers')
            num_features = 767
            num_classes = 10
        elif args.dataset == 'AmazonPhotos':
            dataset = Amazon('./data', 'Photo')
            num_features = 745
            num_classes = 8
        elif args.dataset == 'PPI':
            datasetTrain = PPI('./data', 'train')
            datasetVal = PPI('./data', 'val')
            datasetTest = PPI('./data', 'test')
            num_features = 50
            num_classes = 121

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GATNet(num_features, args.num_hidden_features, args.num_layers, num_classes, args.attention_heads, args.concat_layers, args.model, args.dataset, args.dropout_val).to(device)
        if args.dataset == 'PPI':
            data = datasetTrain[0].to(device)
            dataVal = datasetVal[0].to(device)
            dataTest = datasetTest[0].to(device)
        else:
            data = dataset[0].to(device)
        data = T.NormalizeFeatures()(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)
        if args.verbose:
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
            if args.dataset == 'PPI':
                pred = (out>0.5).int()
                loss = F.cross_entropy(out, data.y)
                correct = (pred == data.y).sum()
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
            if args.use_early_stopping:
                if epoch >= args.num_forced_epochs - 1:
                    model.eval()
                    if args.dataset == 'PPI':
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
                    if acc >= cur_max or loss.item() <= cur_min_loss:
                        if args.verbose:
                            print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                            print('    Old max acc: ' + str(cur_max) + '%')
                            print('    New max acc: ' + str(acc) + '%')
                            print('    Old max loss: ' + str(cur_min_loss) + '%')
                            print('    New max acc: ' + str(loss.item()) + '%')
                            print('')
                        if acc >= cur_max and loss.item() <= cur_min_loss:
                            torch.save(model.state_dict(), "./model/cur_model.pt")
                        cur_max = max(acc, cur_max)
                        cur_min_loss = min(cur_min_loss, loss.item())
                        stop_counter = 0
                    else:
                        stop_counter = stop_counter + 1
                        if args.verbose:
                            print('Did not do better at epoch ' + str(epoch + 1) + '.')
                            print('    Old max: ' + str(cur_max) + '%')
                            print('    Current score: ' + str(acc) + '%')
                            print('')
                        if stop_counter >= args.early_stopping_patience:
                            if args.verbose:
                                print('Stopping training...')
                            stop_training = True
            else:
                if args.verbose:
                    if not epoch == 0 and (epoch + 1) % args.logging_frequency == 0:
                        model.eval()
                        if args.dataset == 'PPI':
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
                if epoch >= args.num_epochs - 1:
                    stop_training = True
            epoch = epoch + 1
        gen_graph(train_accs, "train_accuracy", i)
        gen_graph(train_losses, "train_losses", i)
        gen_graph(val_accs, "validation_accuracy", i)
        gen_graph(val_losses, "validation_losses", i)
        model.eval()
        if args.use_early_stopping:
            model.load_state_dict(torch.load("./model/cur_model.pt"))
        if args.dataset == 'PPI':
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
    print(f'Total Test Average: {total_avg/args.num_runs}')

if __name__ == '__main__':
    main()
