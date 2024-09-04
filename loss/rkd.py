import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GKDLoss']


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class DistillKLWithoutT2(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKLWithoutT2, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, weights=None):
        log_p_s = F.log_softmax(y_s/self.T, dim=1)
        log_p_t = F.log_softmax(y_t/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        input_size = p_t.size(0)
        if weights is not None:
            loss = (weights * p_t * (log_p_t - log_p_s)).sum() / input_size
        else:
            loss = F.kl_div(log_p_s, p_t, reduction='batchmean')
        return loss


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class GKDLoss(nn.Module):
    def __init__(self):
        super(GKDLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.kd = DistillKLWithoutT2(T=0.1)
        self.l2norm = Normalize(2)

    @staticmethod
    def pdist(e, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = torch.exp(-0.5 * (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps))
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


    def forward(self, t_embeddings, s_embeddings, weights):
        feature_kd_loss = torch.tensor(0.).cuda()
        relation_kd_loss = torch.tensor(0.).cuda()

        count = 0
        for i in range(len(s_embeddings)):
            for j in range(len(t_embeddings)):
                weight = weights[count].unsqueeze(-1)
                t_embed = self.l2norm(t_embeddings[j].detach())
                s_embed = self.l2norm(s_embeddings[i])
                B, d = t_embeddings[j].size()
                mask = torch.eye(B).cuda()

                logits = torch.div(torch.mm(s_embed, t_embed.T), 0.1)
                log_prob = logits - torch.log((torch.exp(logits)).sum(1, keepdim=True))
                log_prob = (mask * log_prob * weight).sum(1)
                feature_kd_loss += - log_prob.mean()

                s_relation = self.pdist(s_embed)
                t_relation = self.pdist(t_embed)
                relation_kd_loss += self.kd(s_relation, t_relation, weight)

                count += 1

        return feature_kd_loss/len(s_embeddings), relation_kd_loss/len(s_embeddings)
    