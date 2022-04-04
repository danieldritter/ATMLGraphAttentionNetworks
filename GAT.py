import torch_geometric
import torch

class GraphAttentionLayer(torch_geometric.nn.MessagePassing):

    def __init__(self, input_channels, output_channels, num_heads=1, concat=False, dropout=0.6):
        super().__init__(aggr='add', node_dim=0)
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

        self.attention_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.concat = concat
        if not concat:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels*num_heads))
        

    def forward(self, x, edge_index):
        # Following the GCN tutorial here, but it makes for nodes to be able to attend to themselves?
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0), fill_value="mean")
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
        return self.propagate(edge_index, x=transformed_nodes, attention_vals=(attention_vals1, attention_vals2)) + self.bias

    def message(self, x_j, attention_vals_j, attention_vals_i, index):
        attention_vals = attention_vals_i + attention_vals_j
        attention_vals = self.attention_relu(attention_vals)
        # using torch_geometrics masked softmax implementation here 
        attention_vals = torch_geometric.utils.softmax(attention_vals, index)
        attention_vals = torch.nn.functional.dropout(attention_vals, p=self.dropout_val, training=self.training)
        out = x_j * attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1)
        if self.concat:
            out = out.reshape(out.shape[0],-1)
        else:
            out = torch.mean(out, dim=1)
        return out
