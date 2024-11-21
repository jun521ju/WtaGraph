import torch as th
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_feats_node, in_feats_edge, out_feats, num_heads=1, activation=None, dropout=0.0, bias=True):
        super(GATLayer, self).__init__()
        self.in_feats_node = in_feats_node
        self.in_feats_edge = in_feats_edge
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout else None

        # Linear transformations for multi-head attention
        self.fc_node = nn.Linear(in_feats_node, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(in_feats_edge, out_feats * num_heads, bias=False)  # Replaced MLP with a single linear layer

        # Attention weights
        self.attn_l = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))

        # Learnable edge weights
        self.edge_weights = nn.Parameter(th.Tensor(out_feats * num_heads, out_feats * num_heads))
        nn.init.xavier_uniform_(self.edge_weights)

        # Bias term
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_edge.weight)  # Updated initialization for the single linear layer
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
        nn.init.xavier_normal_(self.attn_e)
        nn.init.xavier_uniform_(self.edge_weights)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def edge_attention(self, edges):
        """
        Computes unnormalized attention scores based on nodes and edge features.
        """
        # Compute attention from source and destination nodes
        el = (edges.src['z'] * self.attn_l).sum(dim=-1)  # (E, H)
        er = (edges.dst['z'] * self.attn_r).sum(dim=-1)  # (E, H)

        # Transform edge features to align dimensions
        transformed_z_e = edges.data['z_e'].view(-1, self.out_feats * self.num_heads)

        # Use learnable edge weights
        weighted_z_e = transformed_z_e @ self.edge_weights.T
        weighted_z_e = weighted_z_e.view(-1, self.num_heads, self.out_feats)
        ee = (weighted_z_e * self.attn_e).sum(dim=-1)  # (E, H)

        # Aggregate attention scores
        e = F.leaky_relu(el + er + ee)  # (E, H)

        return {'e': e}

    def message_func(self, edges):
        """
        Sends messages along the edges during the message-passing phase.
        """
        return {'z': edges.src['z'], 'e': edges.data['e'], 'z_e': edges.data['z_e']}

    def reduce_func(self, nodes):
        """
        Reduces incoming messages at each node using attention scores.
        """
        alpha = F.softmax(nodes.mailbox['e'], dim=1)  # Compute attention weights
        h = th.sum(alpha.unsqueeze(-1) * nodes.mailbox['z'], dim=1)  # Weighted sum of messages
        return {'h': h}

    def forward(self, g, nf, ef):
        # Ensure ef has correct shape
        ef = ef.view(-1, self.in_feats_edge)

        # Transform node and edge features
        z = self.fc_node(nf).view(-1, self.num_heads, self.out_feats)  # Node features: [num_nodes, num_heads, out_feats]
        z_e = self.fc_edge(ef).view(-1, self.num_heads, self.out_feats)  # Updated to use fc_edge

        g.ndata['z'] = z
        g.edata['z_e'] = z_e

        # Apply attention mechanism
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        # Node and edge outputs
        n_out = g.ndata.pop('h').view(-1, self.num_heads * self.out_feats)
        e_out = g.edata.pop('z_e').view(-1, self.num_heads * self.out_feats)

        # Combine node and edge features
        g.edata['z_e'] = e_out.view(-1, self.num_heads, self.out_feats)
        g.apply_edges(lambda edges: {'fused': edges.src['z'] + edges.data['z_e']})
        fused_out = g.edata.pop('fused')

        # Optional: Apply bias, activation, and dropout
        if self.bias is not None:
            reshaped_bias = self.bias.view(self.num_heads, self.out_feats).unsqueeze(0)  # Align dimensions
            fused_out += reshaped_bias
        if self.activation:
            fused_out = self.activation(fused_out)
        if self.dropout:
            fused_out = self.dropout(fused_out)

        return n_out, fused_out


class WTAGNN(nn.Module):
    def __init__(self, g, input_node_feat_size, input_edge_feat_size, n_hidden, n_classes,
                 n_layers, n_heads, activation, dropout):
        super(WTAGNN, self).__init__()
        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(input_node_feat_size, input_edge_feat_size, n_hidden, n_heads, activation, dropout))
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_hidden, n_heads, activation, dropout))
        # Output layers
        self.node_out_layer = GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_classes, num_heads=1, activation=None,
                                       dropout=dropout)
        self.edge_transform_layer = nn.Linear(n_classes, n_hidden * n_heads)  # New layer to transform ef
        self.edge_out_layer = nn.Linear(n_hidden * n_heads, n_classes)

    def forward(self, g, nf, ef):
        """
        Forward pass for the WTAGNN model.
        Processes graph layers sequentially and computes outputs for nodes and edges.
        """
        for layer in self.layers:
            nf, ef = layer(g, nf, ef)

        # Compute node logits and updated edge features
        n_logits, ef = self.node_out_layer(g, nf, ef)

        # Transform edge features to match edge_out_layer input requirements
        ef = self.edge_transform_layer(ef.view(-1, ef.size(-1)))

        # Compute edge logits
        e_logits = self.edge_out_layer(ef)

        return n_logits, e_logits

