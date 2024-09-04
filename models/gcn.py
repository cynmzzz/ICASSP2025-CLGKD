import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['GCN']

class GCN(nn.Module):
    def __init__(self, input_plane, output_plane):
        super(GCN, self).__init__()
        inter_plane = input_plane // 2
        self.node_k = nn.Linear(input_plane, inter_plane, bias=False)
        self.node_v = nn.Linear(input_plane, inter_plane, bias=False)
        self.node_q = nn.Linear(input_plane, inter_plane, bias=False)


        self.conv1 = nn.Sequential(nn.Linear(inter_plane, input_plane, bias=False),
                                   nn.BatchNorm1d(input_plane),
                                   nn.ReLU())

        self.out_conv = nn.Sequential(nn.Linear(input_plane * 2, output_plane, bias=False),
                                      nn.BatchNorm1d(output_plane),
                                      nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c = node_k.size()

        node_q = node_q.view(b, c)
        node_k = node_k.view(b, c).permute(1, 0)
        node_v = node_v.view(b, c)

        sim_map = torch.mm(node_q, node_k)
        normalized_sim_map = self.softmax(sim_map) * (c**-.5)

        gcn_feats = torch.mm(normalized_sim_map, node_v)
        gcn_feats = self.conv1(gcn_feats)

        out_feats = torch.cat([x, gcn_feats], dim=1)
        out_feats = self.out_conv(out_feats)
        return out_feats