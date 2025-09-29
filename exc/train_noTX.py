import torch
import torch.nn as nn
from dataset.dataset_noTx import Dataset_noTX, InferDataset_noTX
from torch.utils.data import DataLoader
from model.eesm import LinearEESM, QformerM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, f1_score
)
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

warnings.filterwarnings("ignore")
def id_from_paths(paths):
    if isinstance(paths, list):
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        return '+'.join(names)
    return os.path.splitext(os.path.basename(paths))[0]


class Train_noTX():
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = self._get_dataset()
        self.rate_to_label = self.dataset.rate_to_label
        self.label_to_rate = self.dataset.label_to_rate
        self.rate_list = self.dataset.rate_list
        self.mean_rate, self.var_rate = self.dataset.mean_rate, self.dataset.var_rate
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
        return Dataset_noTX(paths,
                            ref_stats={
                                "rate_list": self.rate_list,
                                "mean_rate": self.mean_rate,
                                "var_rate": self.var_rate,
                                "rate_to_label": self.rate_to_label
                            })

    def _get_infer_dataset(self):
        paths = self.cfg.data.infer_dataset_paths
        return InferDataset_noTX(paths)




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
        best_r2 = 0
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

            if r2 > best_r2:
                os.makedirs(f'./checkpoints/{self.cfg.data.train_dataset_paths}/{self.cfg.model.model_name}/', exist_ok=True)
                best_r2 = r2
                torch.save(self.model.state_dict(), f'./checkpoints/{self.cfg.data.train_dataset_paths}/{self.cfg.model.model_name}/best_model.pth')


    def test(self):
        self.test_dataset = self._get_test_dataset()
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.test.batch_size, shuffle=False)

        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f'./checkpoints/{self.cfg.data.train_dataset_paths}/{self.cfg.model.model_name}/best_model.pth'))

        self.model.eval()

        result_dir = os.path.join(
            "result", "test",
            f"{id_from_paths(self.cfg.data.train_dataset_paths)}2{id_from_paths(self.cfg.data.test_dataset_paths)}",
            self.cfg.model.model_name
        )
        os.makedirs(result_dir, exist_ok=True)

  
        pred_rate = []
        pred_target = []
        true_target = []
        true_label = []

        for i, batch in tqdm(enumerate(self.test_dataloader)):
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

        acc = accuracy_score(true_label, pred_label)
        recall_macro = recall_score(true_label, pred_label, average='macro', zero_division=0)
        f1_macro = f1_score(true_label, pred_label, average='macro', zero_division=0)

        # 反归一化
        true_target = true_target * self.dataset.var_rate + self.dataset.mean_rate
        pred_rate = pred_rate * self.dataset.var_rate + self.dataset.mean_rate
        pred_target = pred_target * self.dataset.var_rate + self.dataset.mean_rate

        mse = mean_squared_error(true_target, pred_rate)
        mae = mean_absolute_error(true_target, pred_rate)
        r2 = r2_score(true_target, pred_target)


        print(f"TEST MSE: {mse}, MAE: {mae}, R2: {r2}, ACC: {acc}, RECALL(M): {recall_macro}, F1(M): {f1_macro}")


        # true_label_rate = [self.label_to_rate[label] for label in true_label]
        # pred_label_rate = [self.label_to_rate[label] for label in pred_label]

        label_ids = sorted(self.label_to_rate.keys())
        cm = confusion_matrix(true_label, pred_label, labels=label_ids)
        display_labels = [f"{self.label_to_rate[i]:.1f}" for i in label_ids] 

        fig_cm, ax_cm = plt.subplots(figsize=(8,8), dpi=300)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=True, values_format='d')
        ax_cm.set_title("Confusion Matrix")
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(result_dir, "confusion_matrix.pdf"), format="pdf")
        plt.close(fig_cm)

        residuals = pred_rate - pred_target
        fig_res, ax_res = plt.subplots(figsize=(6, 4), dpi=300)
        ax_res.scatter(true_target, residuals, s=10, alpha=0.8)
        ax_res.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax_res.set_xlabel("True Value")
        ax_res.set_ylabel("Residual (Pred - True)")
        ax_res.set_title("Residuals vs True")
        fig_res.tight_layout()
        fig_res.savefig(os.path.join(result_dir, "residuals_vs_true.pdf"), format="pdf")
        plt.close(fig_res)

        with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R2: {r2:.6f}\n")
            f.write(f"ACC: {acc:.6f}\n")
            f.write(f"RECALL_macro: {recall_macro:.6f}\n")
            f.write(f"F1_macro: {f1_macro:.6f}\n")
            f.write("Confusion Matrix (rows=true, cols=pred):\n")
            for row in cm:
                f.write(" ".join(str(int(v)) for v in row) + "\n")




    def infer(self):
        self.infer_dataset = self._get_infer_dataset()
        self.infer_dataloader = DataLoader(self.infer_dataset, batch_size=self.cfg.test.batch_size, shuffle=False)
        
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f'./checkpoints/{self.cfg.data.train_dataset_paths}/{self.cfg.model.model_name}/best_model.pth'))

        self.model.eval()

        infer_paths = self.cfg.data.infer_dataset_paths
        base_path = infer_paths[0] if isinstance(infer_paths, list) else infer_paths
        output_dir = os.path.dirname(base_path)
        infer_name = os.path.splitext(os.path.basename(base_path))[0]
        output_file = os.path.join(output_dir, f"问题2-{infer_name}.xlsx")
        

  
        pred_rate = []
        pred_target = []


        for i, batch in tqdm(enumerate(self.infer_dataloader)):
            feature, target, label = batch
            feature = feature.to(self.device)
            rate = self.model.cacluate_rate(feature, torch.tensor(self.rate_list))
            pred_rate.extend(rate.cpu().detach().numpy())
            pred_target.extend(self.model.get_rateindex(feature, torch.tensor(self.rate_list)))

            # rate = target = self.model.get_rateindex(feature, torch.tensor(self.rate_list))
            # pred_rate.extend(rate)
            # pred_target.extend(target)

        pred_rate = np.array(pred_rate)
        pred_target = np.array(pred_target)

  


        pred_rate = pred_rate * self.dataset.var_rate + self.dataset.mean_rate
        pred_target = pred_target * self.dataset.var_rate + self.dataset.mean_rate

        print(f"unique pred_target: {np.unique(pred_target)}, len: {len(np.unique(pred_target))}")

        # 保存结果
        df = pd.DataFrame({"pred_target": pred_target})
        df.to_excel(output_file, index=False)
        print(f"Saved infer results to: {output_file}")



















        

    


