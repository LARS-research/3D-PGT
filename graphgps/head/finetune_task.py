import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.utils import k_hop_subgraph

@register_head('finetune_task')
class PretrainTask(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # bond_length
        self.bl_reduce_layer = nn.Linear(dim_in * 3, dim_in)
        list_bl_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_bl_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.bl_layers = nn.ModuleList(list_bl_layers)

        # bond_angle
        list_ba_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_ba_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.ba_layers = nn.ModuleList(list_ba_layers)

        # dihedral_angle
        list_da_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_da_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.da_layers = nn.ModuleList(list_da_layers)

        # graph-level prediction (energy)
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)

        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]

    def _apply_index(self, batch):
        return batch.bond_length, batch.distance

    def forward(self, batch):


        # total energy
        graph_rep = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_rep = self.ba_layers[l](graph_rep)
            graph_rep = self.activation(graph_rep)
        graph_rep = self.ba_layers[self.L](graph_rep)

        return graph_rep, batch.y

