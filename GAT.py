import torch_geometric
import torch
import torch.nn.functional as F


class GraphAttentionLayer(torch_geometric.nn.MessagePassing):

    def __init__(self, input_channels, output_channels, num_heads=1, concat=False, dropout=0.6):
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

        # self.attention_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.attention_relu = torch.nn.Softmax()
        self.concat = concat
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_channels, 100),
            torch.nn.PReLU(),
            torch.nn.Linear(100, output_channels)
        )
        if not concat:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels * num_heads))

    def forward(self, x, edge_index):
        # Following the GCN tutorial here, but it makes for nodes to be able to attend to themselves?
        edge_ind, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0), fill_value="mean")
        transformed_nodes = []
        attention_vals1 = []
        attention_vals2 = []
        for i in range(self.num_heads):
            # print(f'input channels: {self.input_channels}')
            # print(f'output channels: {self.output_channels}')
            # print(f'x shape: {x.shape}')
            mlp_x = self.mlp(x)
            # print(f'mlp_x shape: {mlp_x.shape}')

            transform_x = self.ws[i](x)
            # transform_x = mlp_x
            # print(f'transform_x: {transform_x.shape}')
            attention1 = self.attentions1[i](transform_x)
            attention2 = self.attentions2[i](transform_x)
            transformed_nodes.append(transform_x)
            attention_vals1.append(attention1)
            attention_vals2.append(attention2)
        transformed_nodes = torch.stack(transformed_nodes)
        transformed_nodes = torch.transpose(transformed_nodes, 0, 1)
        # print(f'transformed_nodes: {transformed_nodes.shape}')
        attention_vals1 = torch.stack(attention_vals1).squeeze(-1).T
        attention_vals2 = torch.stack(attention_vals2).squeeze(-1).T
        return self.propagate(edge_ind, x=transformed_nodes,
                              attention_vals=(attention_vals1, attention_vals2)) + self.bias

    def message(self, x_j, attention_vals_j, attention_vals_i, index):
        # print(f'x_j size: {x_j.shape}')
        attention_vals = attention_vals_i + attention_vals_j
        # print(f'attention_vals size: {attention_vals.shape}')
        attention_vals = self.attention_relu(attention_vals)
        # using torch_geometrics masked softmax implementation here 
        attention_vals = torch_geometric.utils.softmax(attention_vals, index)
        attention_vals = torch.nn.functional.dropout(attention_vals, p=self.dropout_val, training=self.training)
        # print(f'attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1) size: {attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1).shape}')
        out = x_j * attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1)
        # print(f'out shape: {out.shape}')
        if self.concat:
            out = out.reshape(out.shape[0], -1)
            # print(out.shape)
        else:
            out = torch.mean(out, dim=1)
        return out


class EGAT_Layer(torch_geometric.nn.MessagePassing):
    def __init__(self, input_nodes, input_edges, output_nodes, output_edges, num_heads=1, concat=False, dropout=0.6):
        super().__init__(aggr='add', node_dim=0)
        self.input_nodes = input_nodes
        self.input_edges = input_edges
        self.output_nodes = output_nodes
        self.num_heads = num_heads
        self.dropout_val = dropout

        # Creating attention mechanism in two layers to compute values at node level
        # Avoids having to separately deal with every pairing of nodes
        self.ws = torch.nn.ModuleList()
        self.we = torch.nn.ModuleList()

        self.attentions1 = torch.nn.ModuleList()
        self.attentions2 = torch.nn.ModuleList()
        self.edge_att = torch.nn.ModuleList()
        for i in range(num_heads):
            head_transform = torch.nn.Linear(input_nodes, output_nodes)
            edge_transform = torch.nn.Linear(input_edges, output_edges)
            attention1 = torch.nn.Linear(output_nodes, 1)
            attention2 = torch.nn.Linear(output_nodes, 1)
            edge_attention = torch.nn.Linear(output_edges, 1)
            torch.nn.init.xavier_uniform_(head_transform.weight)
            torch.nn.init.xavier_uniform_(attention1.weight)
            torch.nn.init.xavier_uniform_(attention2.weight)
            torch.nn.init.xavier_uniform_(edge_attention.weight)
            self.ws.append(head_transform)
            self.we.append(edge_transform)
            self.attentions1.append(attention1)
            self.attentions2.append(attention2)
            self.edge_att.append(edge_attention)

        self.attention_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.concat = concat
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(input_nodes, 100),
        #     torch.nn.PReLU(),
        #     torch.nn.Linear(100, output_classes)
        # )
        if not concat:
            self.bias = torch.nn.Parameter(torch.zeros(output_nodes + output_edges))
        else:
            self.bias = torch.nn.Parameter(torch.zeros((output_nodes + output_edges) * num_heads))

    def forward(self, x, edge_index, edge_features):
        # Following the GCN tutorial here, but it makes for nodes to be able to attend to themselves?
        edge_ind, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0), fill_value="mean")

        # Modify the Edge features to a N * N * F_e Matrix
        edge_f = edge_features

        transformed_nodes = []
        transformed_edges = []
        attention_vals1 = []
        attention_vals2 = []
        edge_attention = []

        for i in range(self.num_heads):
            # print(f'input channels: {self.input_nodes}')
            # print(f'output channels: {self.output_nodes}')
            # print(f'x shape: {x.shape}')
            # print(f'mlp_x shape: {mlp_x.shape}')
            # transform_x = mlp_x
            # print(f'transform_x: {transform_x.shape}')

            transform_x = self.ws[i](x)
            transfored_e = self.we[i](edge_f)
            attention1 = self.attentions1[i](transform_x)
            attention2 = self.attentions2[i](transform_x)
            edge_att = self.edge_att[i](transfored_e)
            transformed_nodes.append(transform_x)
            transformed_edges.append(transfored_e)
            attention_vals1.append(attention1)
            attention_vals2.append(attention2)
            edge_attention.append(edge_att)

        transformed_nodes = torch.stack(transformed_nodes)
        transformed_nodes = torch.transpose(transformed_nodes, 0, 1)
        transformed_edges = torch.stack(transformed_edges)
        transformed_edges = torch.transpose(transformed_edges, 0, 1)
        # print(f'transformed_nodes: {transformed_nodes.shape}')
        # print(f'transformed_edges: {transformed_edges.shape}')

        attention_vals1 = torch.stack(attention_vals1).squeeze(-1).T
        attention_vals2 = torch.stack(attention_vals2).squeeze(-1).T
        # print(f'attention_vals1 size: {attention_vals1.shape}')
        # print(f'attention_vals2 size: {attention_vals2.shape}')
        edge_attention = torch.stack(edge_attention).squeeze(-1).T

        # print(transformed_edges.isEmpty)
        return self.propagate(edge_index,
                              x=transformed_nodes,
                              edge_feature=transformed_edges,
                              attentions_edge=edge_attention,
                              attention_vals=(attention_vals1, attention_vals2),
                              ) + self.bias

    def message(self, x_j, edge_feature, attentions_edge, attention_vals_j, attention_vals_i, index):
        # print(f'x_j size: {x_j.shape}')
        # print(f'edge_feature_j size: {edge_feature.shape}')
        attention_vals = attention_vals_i + attention_vals_j
        # print(f'attention_vals size: {attention_vals.shape}')
        attention_vals = self.attention_relu(attention_vals)
        # using torch_geometrics masked softmax implementation here
        attention_vals = torch_geometric.utils.softmax(attention_vals, index)
        attention_vals = torch.nn.functional.dropout(attention_vals, p=self.dropout_val, training=self.training)
        # print(f'attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1) size: {attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1).shape}')

        edge_attention = attentions_edge
        # print(f'edge_attention size: {edge_attention.shape}')
        # print(f'edge_attention size: {edge_attention.shape}')
        edge_attention = self.attention_relu(edge_attention)
        # print(f'index size: {index.shape}')
        edge_attention = torch_geometric.utils.softmax(edge_attention, index)
        edge_attention = torch.nn.functional.dropout(edge_attention, p=self.dropout_val, training=self.training)

        out_node = x_j * attention_vals.view(attention_vals.shape[0], attention_vals.shape[1], 1)
        out_edge = edge_feature * edge_attention.view(edge_attention.shape[0], edge_attention.shape[1], 1)
        out = torch.concat((out_node, out_edge), dim=-1)
        # print(f'out_node size: {out_node.shape}')
        # print(f'out_edge size: {out_edge.shape}')
        # print(f'out size: {out.shape}')
        if self.concat:
            out = out.reshape(out.shape[0], -1)
            # print(f'out size: {out.shape}')
        else:
            out = torch.mean(out, dim=1)
        return out
