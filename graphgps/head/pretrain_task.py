import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.utils import k_hop_subgraph

@register_head('pretrain_task')
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
        # bond_length
        bond_length_pair = batch.positions[batch.edge_index.T]
        bond_length_true = torch.sum((bond_length_pair[:, 0, :] - bond_length_pair[:, 1, :]) ** 2, axis=1)
        bond_length_pred = torch.concat((batch.x[batch.edge_index.T][:,0,:], batch.x[batch.edge_index.T][:,1,:], batch.edge_attr),axis=1)
        bond_length_pred = self.bl_reduce_layer(bond_length_pred)
        for l in range(self.L + 1):
            bond_length_pred = self.activation(bond_length_pred)
            bond_length_pred = self.bl_layers[l](bond_length_pred)

        # bond_angle
        bond_angle_pred = batch.x
        for l in range(self.L):
            bond_angle_pred = self.ba_layers[l](bond_angle_pred)
            bond_angle_pred = self.activation(bond_angle_pred)
        bond_angle_pred = self.ba_layers[self.L](bond_angle_pred)

        # dihedral_angle
        dihedral_angle_pred = batch.edge_attr
        for l in range(self.L):
            dihedral_angle_pred = self.da_layers[l](dihedral_angle_pred)
            dihedral_angle_pred = self.activation(dihedral_angle_pred)
        dihedral_angle_pred = self.da_layers[self.L](dihedral_angle_pred)

        # total energy
        graph_rep = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_rep = self.ba_layers[l](graph_rep)
            graph_rep = self.activation(graph_rep)
        graph_rep = self.ba_layers[self.L](graph_rep)

        return bond_length_pred, bond_length_true, bond_angle_pred, batch.bond_angle_true, dihedral_angle_pred, batch.dihedral_angle_true, graph_rep, batch.energy

