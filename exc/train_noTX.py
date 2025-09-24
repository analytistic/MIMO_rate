import torch
import torch.nn as nn
from dataset.dataset_noTx import Dataset_noTX
from torch.utils.data import DataLoader
from model.eesm import LinearEESM, QformerM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings("ignore")



class Train_noTX():
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = self._get_dataset()
        self.rate_to_label = self.dataset.rate_to_label
        self.rate_list = self.dataset.rate_list
        print(f"Rate list: {self.rate_list}")
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfg.train.batch_size, shuffle=True)
        self.model = self._get_model()
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.learning_rate)


    def _get_model(self):
        model_name = self.cfg.model.model_name
        map_model = {
            'LinearEESM': LinearEESM,
            'QformerM': QformerM
        }
        return map_model[model_name](self.cfg, rate_list=self.rate_list)





    def _get_dataset(self):
        paths = self.cfg.data.train_dataset_paths
        return Dataset_noTX(paths)
    

    def _get_test_dataset(self):
        paths = self.cfg.data.test_dataset_paths
        return Dataset_noTX(paths)

    def vali(self):
        # 我们在这个训练集上验证
        self.model.eval()
  
        pred_rate = []
        pred_target = []
        true_target = []
        true_label = []

        for i, batch in tqdm(enumerate(self.dataloader)):
            feature, target, label = batch
            feature = feature.to(self.device)
            target = target.to(self.device)
            target = torch.tensor(target).to(self.device)
            rate = self.model.cacluate_rate(feature, torch.tensor(self.rate_list))
            pred_rate.extend(rate.cpu().detach().numpy())
            pred_target.extend(self.model.get_rateindex(feature, torch.tensor(self.rate_list)))
            true_target.extend(target.cpu().numpy())
            true_label.extend(label.cpu().numpy())
            # rate = target = self.model.get_rateindex(feature, torch.tensor(self.rate_list))
            # pred_rate.extend(rate)
            # pred_target.extend(target)

          


        pred_rate = np.array(pred_rate)
        pred_target = np.array(pred_target)
        true_target = np.array(true_target)
        true_label = np.array(true_label, dtype=int)
        pred_label = np.array([self.rate_to_label[np.float16(rate)] for rate in pred_target], dtype=int)

        acc = np.sum(pred_label == true_label)/len(pred_label)

        # 反归一化
        true_target = true_target * self.dataset.var_rate + self.dataset.mean_rate
        pred_rate = pred_rate * self.dataset.var_rate + self.dataset.mean_rate
        pred_target = pred_target * self.dataset.var_rate + self.dataset.mean_rate

        mse = mean_squared_error(true_target, pred_rate)
        mae = mean_absolute_error(true_target, pred_rate)
        r2 = r2_score(true_target, pred_target)


        return mse, mae, r2, acc








    

    def train(self):
        self.model.to(self.device)
        self.model.train()
        best_acc = 0
        for epoch in range(self.cfg.train.epochs):
            epoch_loss = []
            for i, batch in tqdm(enumerate(self.dataloader)):
                
                feature, target, label = batch
                feature = feature.to(self.device)
                target = target.to(self.device)
                loss = self.model(feature, target, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())

            mse, mae, r2, acc = self.vali()

            print(f"Epoch {epoch+1}/{self.cfg.train.epochs}, Loss: {sum(epoch_loss)/len(epoch_loss)}")
            print(f"EVAL MSE: {mse}, MAE: {mae}, R2: {r2}, ACC: {acc}")

            if acc > best_acc:
                os.makedirs(f'./checkpoints/{self.cfg.model.model_name}', exist_ok=True)
                best_acc = acc
                torch.save(self.model.state_dict(), f'./checkpoints/{self.cfg.model.model_name}/best_model.pth')


    def test(self):
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f'./checkpoints/{self.cfg.model.model_name}/best_model.pth'))
        self._get_test_dataset()
        self.model.eval()

  
        pred_rate = []
        pred_target = []
        true_target = []
        true_label = []

        for i, batch in tqdm(enumerate(self.dataloader)):
            feature, target, label = batch
            feature = feature.to(self.device)
            target = target.to(self.device)
            target = torch.tensor(target).to(self.device)
            rate = self.model.cacluate_rate(feature, torch.tensor(self.rate_list))
            pred_rate.extend(rate.cpu().detach().numpy())
            pred_target.extend(self.model.get_rateindex(feature, torch.tensor(self.rate_list)))
            true_target.extend(target.cpu().numpy())
            true_label.extend(label.cpu().numpy())
            # rate = target = self.model.get_rateindex(feature, torch.tensor(self.rate_list))
            # pred_rate.extend(rate)
            # pred_target.extend(target)

          


        pred_rate = np.array(pred_rate)
        pred_target = np.array(pred_target)
        true_target = np.array(true_target)
        true_label = np.array(true_label, dtype=int)
        pred_label = np.array([self.rate_to_label[np.float16(rate)] for rate in pred_target], dtype=int)

        acc = np.sum(pred_label == true_label)/len(pred_label)

        # 反归一化
        true_target = true_target * self.dataset.var_rate + self.dataset.mean_rate
        pred_rate = pred_rate * self.dataset.var_rate + self.dataset.mean_rate
        pred_target = pred_target * self.dataset.var_rate + self.dataset.mean_rate

        mse = mean_squared_error(true_target, pred_rate)
        mae = mean_absolute_error(true_target, pred_rate)
        r2 = r2_score(true_target, pred_target)


        print(f"TEST MSE: {mse}, MAE: {mae}, R2: {r2}, ACC: {acc}")







        

    


