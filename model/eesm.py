
from torch import nn
import torch
from utils.losses import Losses
from module.qformer import Qformer

class LinearEESM(nn.Module):
    def __init__(self, cfg, rate_list):
        super(LinearEESM, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.k_1 = nn.Parameter(torch.tensor(1.0))
        self.k_2 = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.criterion = Losses(cfg.loss, rate_list=rate_list)
        self.batch_norm = nn.BatchNorm1d(122)
    
    def Phi(self, x):
        y = 1 - torch.exp(- x)
        return y
    
    def InvPhi(self, y):
        x = - torch.log(1 - y + 1e-4)
        return x
    


    def forward(self, x, target, label):
        # x: B, 122
        # x = torch.log(x)
        x = self.batch_norm(x)
        # sinr_eff = self.alpha * self.InvPhi(torch.mean(self.Phi(x/(self.beta+1e-8)), dim=-1))
        sinr_eff = torch.mean(x, dim=-1)


        rate = self.k_1 * sinr_eff + self.bias

        loss = self.criterion(rate, target)
        print(self.alpha.item(), self.beta.item(), self.k_1.item(), self.bias.item())
        return loss



    def cacluate_sinr_eff(self, x):
        # x = torch.log(x)
        x = self.batch_norm(x)
        # sinr_eff = self.alpha * self.InvPhi(torch.mean(self.Phi(x/(self.beta+1e-8)), dim=-1))
        sinr_eff = torch.mean(x, dim=-1)
        return sinr_eff
    
    def cacluate_rate(self, x):
        rate = self.k_1 * self.cacluate_sinr_eff(x) + self.bias
        if torch.any(torch.isinf(rate)):
            raise ValueError("Inf values found in rate")    
        
        return rate
    
    def get_rateindex(self, x, rate_list):
        # rate_list: num x
        # x: B, 122
        rate = self.cacluate_rate(x)
        rate_list = rate_list.to(rate.device)
        rate_list = rate_list.unsqueeze(0).expand(rate.shape[0], -1)  # B, num
        rate_index = torch.argmin(torch.abs(rate.unsqueeze(1) - rate_list), dim=1) # B,
        dec_rate = rate_list[0, rate_index]  # B,



        
        return dec_rate.detach().cpu().numpy()
    

class QformerM(nn.Module):
    def __init__(self, cfg, rate_list):
        super(QformerM, self).__init__()
        self.querys = nn.Embedding(cfg.model.qformer.num_embeddings, cfg.model.qformer.embedding_dim)
        self.Wq = nn.Linear(cfg.model.qformer.embedding_dim, cfg.model.qformer.query_dim)
        self.Wv = nn.Linear(8, cfg.model.qformer.hid_dim)
        self.classifier = nn.Linear(cfg.model.qformer.query_dim, 1)
        self.predict = nn.Linear(cfg.model.qformer.query_dim * cfg.model.qformer.num_embeddings, 1)
        self.qformer = Qformer(
            query_dim=cfg.model.qformer.query_dim,
            hid_dim=cfg.model.qformer.hid_dim,
            q_dim=cfg.model.qformer.q_dim,
            v_dim=cfg.model.qformer.v_dim,
            num_heads=cfg.model.qformer.num_heads,
            num_layers=cfg.model.qformer.num_layers
        )
        self.criterion = Losses(cfg.loss, rate_list=rate_list) 
        

    
    


    def forward(self, x, target, label):
        # x: B, 122
        # target: B, 1
        # label: B, 1

        query = self.querys(torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(x.device)) # 10, dim
        query = self.Wq(query).unsqueeze(0).expand(x.shape[0], -1, -1) # B, 10, 8

        x = self.Wv(x)

        query = self.qformer(query, x) # B, 10, dim
        logits = self.classifier(query).squeeze(-1) # B, 10
        loss = self.criterion(logits, label.long())
        # pred_rate = self.predict(query.flatten(start_dim=1)) # B, 1
        # loss = self.criterion(pred_rate, target)

        return loss
    


    
    def cacluate_rate(self, x, rate_list):
        query = self.querys(torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(x.device))
        query = self.Wq(query).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.Wv(x)
        query = self.qformer(query, x)
        # rate = self.predict(query.flatten(start_dim=1)).squeeze(-1) # B
        logits = self.classifier(query).squeeze(-1)
        index = logits.argmax(dim=-1) # B,
        rate = rate_list[index]  # B,

        
        return rate


    
    def get_rateindex(self, x, rate_list):
        # rate_list: num x
        # x: B, 122
        query = self.querys(torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(x.device))
        query = self.Wq(query).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.Wv(x)
        query = self.qformer(query, x)
        logits = self.classifier(query).squeeze(-1)
        index = logits.argmax(dim=-1) # B,
        dec_rate = rate_list[index]  # B,
        
        return dec_rate.detach().cpu().numpy()

    # def get_rateindex(self, x, rate_list):
    #     # rate_list: num x
    #     # x: B, 122
    #     rate = self.cacluate_rate(x)
    #     rate_list = rate_list.to(rate.device)
    #     rate_list = rate_list.unsqueeze(0).expand(rate.shape[0], -1)  # B, num
    #     rate_index = torch.argmin(torch.abs(rate.unsqueeze(1) - rate_list), dim=1) # B,
    #     dec_rate = rate_list[0, rate_index]  # B,



        
    #     return dec_rate.detach().cpu().numpy()


