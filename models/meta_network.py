import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWeightNetwork(nn.Module):
    def __init__(self, s_feat_num, t_feat_num):
        super(LossWeightNetwork, self).__init__()
        self.s_proj = nn.ModuleList([])
        self.t_proj = nn.ModuleList([])
        self.s_feat_num = s_feat_num
        self.t_feat_num = t_feat_num
        for i in range(len(s_feat_num)):
            self.s_proj.append(nn.Linear(s_feat_num[i], 128))
        for j in range(len(t_feat_num)):
            self.t_proj.append(nn.Linear(t_feat_num[j], 128))

    def similarity(self, f_a, f_b):
        f_a = F.normalize(f_a, dim=1)
        f_b = F.normalize(f_b, dim=1)
        sim = (f_a * f_b).sum(dim=1)
        sim = torch.sigmoid(sim)
        return sim
        
    def forward(self, s_feats, t_feats):
        s_projected_feats = []
        t_projected_feats = []
        for i in range(len(s_feats)):
            s_projected_feats.append(self.s_proj[i](s_feats[i]))
        for i in range(len(t_feats)):
            t_projected_feats.append(self.t_proj[i](t_feats[i]))
        
        weights = []
        for k in range(len(s_projected_feats)):
            for z in range(len(t_projected_feats)):
                weights.append(self.similarity(s_projected_feats[k], t_projected_feats[z]))

        return weights 