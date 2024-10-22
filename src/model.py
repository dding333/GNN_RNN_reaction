import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList, GRU
from torch.nn.functional import one_hot

class RNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_size, device):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = GRU(self.input_dim, int(self.hidden_size / 2), batch_first=True, bidirectional=True)
        self.to(device)

    def forward(self, x):
        # Reshape the input and pass it through the GRU
        x = x.reshape(x.size(0), -1, self.input_dim)
        outputs, last_hidden = self.encoder(x)
        
        # Compute the mean across the sequence length dimension
        output = torch.mean(outputs, dim=1)  # Removing keepdim=True
        return output

class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim, rnn_input_dim=None, rnn_hidden_size=None, latent_size=None, device=None):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = ModuleList([])
        self.rnn = None
        if rnn_input_dim and rnn_hidden_size and latent_size:
            self.rnn = RNNEncoder(rnn_input_dim, rnn_hidden_size, latent_size, device)
        
        if gnn in ['gcn', 'gat', 'sage', 'tag']:
            for i in range(n_layer):
                if gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=feature_len if i == 0 else dim,
                                                     out_feats=dim,
                                                     activation=None if i == n_layer - 1 else torch.relu))
                elif gnn == 'gat':
                    num_heads = 16
                    self.gnn_layers.append(GATConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim // num_heads,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=feature_len if i == 0 else dim,
                                                    out_feats=dim,
                                                    activation=None if i == n_layer - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   k=hops))
        elif gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=feature_len, out_feats=dim, k=n_layer))
        else:
            raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph, sequence_input=None):
        # GNN Embedding
        feature = graph.ndata['feature']
        h = one_hot(feature, num_classes=self.feature_len)
        h = torch.sum(h, dim=1, dtype=torch.float)
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == 'gat':
                h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)

        # RNN Embedding (if sequence input is provided)
        if self.rnn is not None and sequence_input is not None:
            sequence_embedding = self.rnn(sequence_input)
            combined_embedding = torch.cat([graph_embedding, sequence_embedding], dim=-1)
            return combined_embedding
        
        return graph_embedding
