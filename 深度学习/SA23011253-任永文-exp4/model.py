from locale import normalize
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv


class GCNConv(nn.Module):
    def __init__(self, in_channels,out_channels, device):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weight)
        self.device=device
    def forward(self, input, adj_matrix):
        if isinstance(adj_matrix,SparseTensor):
            adj_matrix = adj_matrix.to_dense()
        degree_matrix = torch.diag(torch.sum(adj_matrix, 1)).inverse().sqrt().to(self.device)
        adj_hat_matrix = torch.spmm(torch.spmm(degree_matrix, adj_matrix + torch.eye(adj_matrix.size(0)).to(self.device)), degree_matrix)
        output = torch.relu(torch.spmm(adj_hat_matrix, torch.spmm(input, self.weight)))
        return output


class Net(torch.nn.Module):
    def __init__(self, nfeat, nhid, nout, num_layers, alpha, add_self_loops, normalize, dropout):
        super().__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(nhid, alpha, add_self_loops=add_self_loops, normalize=normalize))
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj_t):
        x = x_0 = self.drop(self.relu(self.fc1(x)))
        for conv in self.convs:
            x = self.drop(self.relu(conv(x, x_0, adj_t) + x))
        x = self.drop(self.relu(self.fc2(x)))
        return x


class Net2(nn.Module):
    def __init__(self, nfeat, nhid, nout, device):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid, device)
        self.conv2 = GCNConv(nhid, nout, device)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = self.conv2(x, adj_t)
        return x