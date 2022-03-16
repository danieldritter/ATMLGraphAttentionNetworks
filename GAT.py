import torch_geometric
import torch

class GraphAttentionLayer(torch_geometric.nn.MessagePassing):

    def __init__(self, input_channels, output_channels, num_heads=1, concat=False):
        super().__init__(aggr='add')
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.linear = torch.nn.Linear(input_channels, output_channels*num_heads)
        # Creating attention mechanism in two layers to compute values at node level
        # Avoids having to separately concatenate nodes and compute values
        self.attention1 = torch.nn.Linear(output_channels*num_heads, num_heads)
        self.attention2 = torch.nn.Linear(output_channels*num_heads, num_heads)
        self.attention_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.concat = concat
        torch.nn.init.xavier_uniform_(self.attention1.weight)
        torch.nn.init.xavier_uniform_(self.attention2.weight)
        if not concat:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels*num_heads))

    def forward(self, x, edge_index):
        # Following the GCN tutorial here, but it makes for nodes to be able to attend to themselves?
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))
        transformed_nodes = self.linear(x).view(-1, self.output_channels*self.num_heads)
        attention_vals1 = self.attention1(transformed_nodes).view(-1, self.num_heads)
        attention_vals2 = self.attention2(transformed_nodes).view(-1, self.num_heads)
        transformed_nodes = transformed_nodes.view(-1, self.output_channels*self.num_heads)
        return self.propagate(edge_index, x=transformed_nodes, attention_vals=(attention_vals1, attention_vals2)) + self.bias

    # Just a weighted sum of attention values and node features
    # Not sure if this should be x_i or x_j
    def message(self, x_j, attention_vals_i, attention_vals_j):
        attention_vals = (attention_vals_i + attention_vals_j)
        attention_vals = self.attention_relu(attention_vals)
        attention_vals = torch.nn.functional.softmax(attention_vals, dim=1)
        attention_vals = torch.nn.functional.dropout(attention_vals, p=0.6, training=self.training)
        # drop_x_j = torch.nn.functional.dropout(x_j, p=0.6, training=self.training)
        drop_x_j = x_j
        drop_x_j = drop_x_j.view(-1, self.output_channels, self.num_heads)
        
        if self.concat:
            out_vals = []
        else:
            avg_x_j = torch.zeros((drop_x_j.shape[0], drop_x_j.shape[1]), device=drop_x_j.device)

        for i in range(self.num_heads):
            if self.concat:
                out_vals.append((drop_x_j[:, :, i].T*attention_vals[:,i]).T)
            else:
                avg_x_j += (drop_x_j[:, :, i].T*attention_vals[:,i]).T
        if self.concat:
            out = torch.cat(out_vals, dim=1)
        else:
            out = avg_x_j / self.num_heads
        return out
