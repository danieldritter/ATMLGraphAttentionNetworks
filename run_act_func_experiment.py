import pickle
import torch
import numpy as np
import torch_geometric
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

"""
Runs an experiment testing different activation functions on the GAT layer
"""

class GraphAttentionLayerActivationTest(torch_geometric.nn.MessagePassing):

    def __init__(self, input_channels, output_channels, num_heads=1, concat=False, dropout=0.6, activation_function=torch.nn.LeakyReLU(negative_slope=0.2)):
        super().__init__(aggr='add', node_dim=0)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.dropout_val = dropout
        # Creating attention mechanism in two layers to compute values at node level
        # Avoids having to separately deal with every pairing of nodes
        self.ws = torch.nn.ModuleList()
        self.attentions1 = torch.nn.ModuleList()
        self.attentions2 = torch.nn.ModuleList()
        for i in range(num_heads):
            head_transform = torch.nn.Linear(input_channels, output_channels)
            attention1 = torch.nn.Linear(output_channels, 1)
            attention2 = torch.nn.Linear(output_channels, 1)
            torch.nn.init.xavier_uniform_(head_transform.weight)
            torch.nn.init.xavier_uniform_(attention1.weight)
            torch.nn.init.xavier_uniform_(attention2.weight)
            self.ws.append(head_transform)
            self.attentions1.append(attention1)
            self.attentions2.append(attention2)

        self.attention_relu = activation_function
        self.concat = concat
        if not concat:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels * num_heads))

    def forward(self, x, edge_index):
        edge_ind, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0), fill_value="mean")
        transformed_nodes = []
        attention_vals1 = []
        attention_vals2 = []
        for i in range(self.num_heads):
            transform_x = self.ws[i](x)
            attention1 = self.attentions1[i](transform_x)
            attention2 = self.attentions2[i](transform_x)
            transformed_nodes.append(transform_x)
            attention_vals1.append(attention1)
            attention_vals2.append(attention2)
        transformed_nodes = torch.stack(transformed_nodes)
        transformed_nodes = torch.transpose(transformed_nodes, 0, 1)
        attention_vals1 = torch.stack(attention_vals1).squeeze(-1).T
        attention_vals2 = torch.stack(attention_vals2).squeeze(-1).T
        return self.propagate(edge_ind, x=transformed_nodes,
                              attention_vals=(attention_vals1, attention_vals2)) + self.bias

    def message(self, x_j, attention_vals_j, attention_vals_i, index):
        attention_vals = attention_vals_i + attention_vals_j
        attention_vals = self.attention_relu(attention_vals)
        # using torch_geometrics masked softmax implementation here
        attention_vals = torch_geometric.utils.softmax(attention_vals, index)
        attention_vals = torch.nn.functional.dropout(attention_vals, p=self.dropout_val, training=self.training)
        out = x_j * attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1)
        if self.concat:
            out = out.reshape(out.shape[0], -1)
        else:
            out = torch.mean(out, dim=1)
        return out

class GATModel(torch.nn.Module):

    def __init__(self, num_input_features, num_output_features, num_heads, num_classes, activation_function=torch.nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.conv1 = GraphAttentionLayerActivationTest(num_input_features, num_output_features, num_heads=num_heads, concat=True, activation_function=activation_function)
        self.conv2 = GraphAttentionLayerActivationTest(num_output_features*num_heads, num_classes, num_heads=1, activation_function=activation_function)

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
    activation_functions = {"log_sigmoid":torch.nn.LogSigmoid(),"tanh":torch.nn.Tanh(),"softmax":torch.nn.Softmax()}
    all_avgs = {}
    for activation in activation_functions:
        activation_function = activation_functions[activation]
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
            model = GATModel(num_features,8,8,num_classes, activation_function=activation_function).to(device)
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
        all_avgs[activation] = f"{avg_acc} +/- {ci}"
    for activation in activation_functions:
        print(f"Activation Function: {activation} Test Accuracy: {all_avgs[activation]}")

if __name__ == '__main__':
    main()
