import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=1.2, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)

        """
        #print('targets:',targets) #batchid*batchimage
        #print('inputs:',inputs,'len:',len(inputs),'shape:',inputs.shape) #batchid*batchimage
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # if len(mask) < 20:
        #     loss = torch.tensor(0)
        #     #loss = torch.tensor(sum(loss_his)*1.2/len(loss_his))
        #     return loss
        #else:
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == False].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        # print('dist_ap:',dist_ap,'len:',len(dist_ap),'shape:',dist_ap.shape)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)#max(0,-y(x1-x2)+marigin )
        #print(loss)
        if self.mutual:
            return loss, dist
        return loss

#addmm  triplet test
# a = torch.randint(1,10,(8,3))
# b = a
# a = torch.pow(a,2).sum(dim=1,keepdim=True)
# a = a.expand(8,8)
# a = a+a.t()
# c = a.addmm(beta=1,alpha=-2,mat1=b,mat2= b.t())
# print(c)
# target = torch.tensor([0,0,1,0,2,1,2,2])
# target = target.expand(8,8)
# print(target)
# mask = target.eq(target.t())
# print(mask)
# dist_ap, dist_an = [], []
# for i in range(8):
#     dist_ap.append(c[i][mask[i]].max().unsqueeze(0))
#     dist_an.append(c[i][mask[i] == 0].min().unsqueeze(0))
# dist_ap = torch.cat(dist_ap)
# dist_an = torch.cat(dist_an)
# print(dist_ap)
# print(dist_an)