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
        self.fc_edge = nn.Linear(in_feats_edge, out_feats * num_heads, bias=False)

        # Attention weights
        self.attn_l = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))

        # Bias term
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_edge.weight)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
        nn.init.xavier_normal_(self.attn_e)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def edge_attention(self, edges):
        # Compute unnormalized attention scores for each head
        el = (edges.src['z'] * self.attn_l).sum(dim=-1)  # Shape: (E, H)
        er = (edges.dst['z'] * self.attn_r).sum(dim=-1)  # Shape: (E, H)
        z_e = edges.data['z_e'].clone()  # Clone to avoid in-place conflicts
        ee = (z_e * self.attn_e).sum(dim=-1)  # Use cloned tensor

        # Combine attention scores
        e = F.leaky_relu(el + er + ee)  # Shape: (E, H)
        print(f"el shape: {el.shape}, er shape: {er.shape}, ee shape: {ee.shape}, e shape: {e.shape}")
        return {'e': e}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'z_e': edges.data['z_e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = th.sum(alpha.unsqueeze(-1) * nodes.mailbox['z'], dim=1)
        return {'h': h}


    def forward(self, g, nf, ef):
        z = self.fc_node(nf).view(-1, self.num_heads, self.out_feats)
        z_e = self.fc_edge(ef).view(-1, self.num_heads, self.out_feats)
        g.ndata['z'] = z
        g.edata['z_e'] = z_e

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        n_out = g.ndata.pop('h').view(-1, self.num_heads * self.out_feats)
        e_out = g.edata.pop('z_e').view(-1, self.num_heads * self.out_feats)

        if self.bias is not None:
            n_out += self.bias.unsqueeze(0)
            e_out += self.bias.unsqueeze(0)

        if self.activation:
            n_out = self.activation(n_out)
            e_out = self.activation(e_out)

        if self.dropout:
            n_out = self.dropout(n_out)
            e_out = self.dropout(e_out)

        return n_out, e_out


class WTAGNN(nn.Module):
    def __init__(self, g, input_node_feat_size, input_edge_feat_size, n_hidden, n_classes,
                 n_layers, n_heads, activation, dropout):
        super(WTAGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(input_node_feat_size, input_edge_feat_size, n_hidden, n_heads, activation, dropout))
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_hidden, n_heads, activation, dropout))
        self.node_out_layer = GATLayer(n_hidden * n_heads, n_hidden * n_heads, n_classes, num_heads=1, activation=None, dropout=dropout)
        self.edge_out_layer = nn.Linear(n_hidden * n_heads, n_classes)

    def forward(self, g, nf, ef):
        for layer in self.layers:
            nf, ef = layer(g, nf, ef)
        n_logits, _ = self.node_out_layer(g, nf, ef)
        e_logits = self.edge_out_layer(ef)
        return n_logits, e_logits
