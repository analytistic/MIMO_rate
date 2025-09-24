import torch
import torch.nn.functional as F

class Losses(torch.nn.Module):
    def __init__(self, cfg, rate_list):
        super(Losses, self).__init__()
        self.loss = cfg.loss_name
        self.cfg = cfg
        self.rate_list = rate_list

    def forward(self, pred, target):
        if self.loss == 'mse':
            return self.mse(pred, target)
        elif self.loss == 'l1':
            return self.l1(pred, target)
        elif self.loss == 'margin_l1':
            return self.margin_loss(pred, target)
        elif self.loss == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()(pred, target)
        elif self.loss == 'CE_Dist':
            return self.ce_with_distance(pred, target)
        elif self.loss == 'GaussianCE':
            return self.gaussian_ce(pred, target)
        elif self.loss == 'EMD':
            return self.emd_loss(pred, target)
        else:
            raise NotImplementedError

    def mse(self, pred, target):
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss

    def l1(self, pred, target):
        loss = torch.nn.functional.l1_loss(pred, target)
        return loss

    def margin_loss(self, pred, target):
        margin = self.cfg.margin_loss.margin
        loss = torch.relu(torch.abs(pred - target) - margin).mean()
        return loss

    # 1) CrossEntropy + 距离期望正则
    def ce_with_distance(self, logits, target):
        # logits: [B, C], target: [B] (long)
        ce = F.cross_entropy(logits, target)
        p = F.softmax(logits, dim=1)  # [B, C]
        C = logits.size(1)
        idx = torch.arange(C, device=logits.device).view(1, -1)  # [1, C]
        dist = (idx - target.view(-1, 1).float()).abs()  # [B, C]
        # 归一化距离到[0,1]
        dist = dist / max(C - 1, 1)
        # 超参
      
        lam =  0.5
        alpha =  1.0
        exp_dist = (p * (dist ** alpha)).sum(dim=1).mean()
        return ce + lam * exp_dist

    # 2) 高斯标签平滑交叉熵
    def gaussian_ce(self, logits, target):
        # logits: [B, C], target: [B] (long)
        B, C = logits.size()
        idx = torch.arange(C, device=logits.device).view(1, -1).float()  # [1, C]
        center = target.view(-1, 1).float()  # [B, 1]

        sigma = 0.2
        # 高斯软标签
        y_soft = torch.exp(-0.5 * ((idx - center) / max(sigma, 1e-6)) ** 2)  # [B, C]
        y_soft = y_soft / (y_soft.sum(dim=1, keepdim=True) + 1e-8)
        logp = F.log_softmax(logits, dim=1)
        loss = -(y_soft * logp).sum(dim=1).mean()
        return loss

    # 3) EMD/Wasserstein(基于CDF的L1/L2差)
    def emd_loss(self, logits, target):
        p = F.softmax(logits, dim=1)  # [B, C]
        B, C = p.size()
        t = F.one_hot(target, num_classes=C).float()  # [B, C]
        # 也可把 t 改为高斯软标签以更“宽容”的对角集中
        # 累计分布
        Pcdf = torch.cumsum(p, dim=1)
        Tcdf = torch.cumsum(t, dim=1)
 
        r = 1
        diff = torch.abs(Pcdf - Tcdf)
        if r == 2:
            diff = diff.pow(2)
        loss = diff.sum(dim=1).mean()
        return loss