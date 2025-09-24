import torch
import torch.nn.functional as F

class Losses(torch.nn.Module):
    def __init__(self, cfg, rate_list):
        super(Losses, self).__init__()
        self.loss = cfg.loss_name
        self.cfg = cfg
        self.rate_list = rate_list
        self.rate_var = max(rate_list) - min(rate_list) if rate_list else 1.0

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
        
        if self.loss == 'OrdinalBCE':
            return self.ordinal_bce(pred, target)
        elif self.loss == 'CE_EV':
            return self.ce_ev(pred, target)
        elif self.loss == 'WindowPenalty':
            return self.window_penalty(pred, target)
        elif self.loss == 'CE_DistFocal':
            return self.ce_dist_focal(pred, target)
 
        
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
        centers = torch.tensor(self.rate_list, device=logits.device, dtype=torch.float32)  # [C]
        assert centers.numel() == C

        true_center = centers[target]


        dist = (centers.view(1, -1) - true_center.view(-1, 1)) / self.rate_var



        sigma = 0.02
    
        # 高斯软标签
        y_soft = torch.exp(- (dist**2) / (2 * sigma**2))  # [B, C]

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
    



    def ordinal_bce(self, logits, target):
        # 用现有 [B,C] logits，无需改头
        p = F.softmax(logits, dim=1)                # [B,C]
        P_gt = 1.0 - torch.cumsum(p, dim=1)[:, :-1] # [B,C-1], P(Y>k)
        ks = torch.arange(logits.size(1)-1, device=logits.device).view(1, -1)
        B_bin = (target.view(-1,1) > ks).float()    # [B,C-1]
        P_gt = P_gt.clamp(1e-6, 1-1e-6)
        return F.binary_cross_entropy(P_gt, B_bin)

    def ce_ev(self, logits, target):
        ce = F.cross_entropy(logits, target)
        p = F.softmax(logits, dim=1)
        centers = torch.tensor(self.rate_list, device=logits.device, dtype=torch.float32) # [C]
        ev = (p * centers.view(1,-1)).sum(dim=1)   # 预测分布的期望 mcs
        true = centers[target]                     # 真实 mcs 中心
        reg = F.smooth_l1_loss(ev, true)
        lam = 1.0
        return ce + lam * reg

    def window_penalty(self, logits, target):
        ce = F.cross_entropy(logits, target)
        p = F.softmax(logits, dim=1)
        centers = torch.tensor(self.rate_list, device=logits.device, dtype=torch.float32)
        true = centers[target]
        d = (centers.view(1,-1) - true.view(-1,1)).abs()
        w = 0.2 * max(float(self.rate_var), 1e-6)  # 窗口宽度可调
        pen = (p * (d > w).float()).sum(dim=1).mean()
        lam = 0.5
        return ce + lam * pen

    def ce_dist_focal(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        p = F.softmax(logits, dim=1)
        centers = torch.tensor(self.rate_list, device=logits.device, dtype=torch.float32)
        true = centers[target]
        dist = (centers.view(1,-1) - true.view(-1,1)).abs() / max(float(self.rate_var), 1e-6)
        exp_dist = (p * (dist**2)).sum(dim=1)      # 远距更重罚
        pt = p.gather(1, target.view(-1,1)).clamp_min(1e-8).squeeze(1)
        gamma = 2.0
        focal = (1 - pt).pow(gamma) * ce
        lam = 0.5
        return (focal + lam * exp_dist).mean()
