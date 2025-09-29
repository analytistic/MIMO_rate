from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class Dataset_noTX(Dataset):
    def __init__(self, paths, ref_stats=None):
        self.paths = paths
        
        self._load_and_combine_data()
        if ref_stats is None:
            self.rate_list, self.mean_rate, self.var_rate, self.rate_to_label = self._get_rate_list()
        else:
            # 测试/推理：用训练提供的
            self.rate_list = ref_stats["rate_list"]
            self.mean_rate = ref_stats["mean_rate"]
            self.var_rate = ref_stats["var_rate"]
            self.rate_to_label = ref_stats["rate_to_label"]

        self.label_to_rate = {v: k for k, v in self.rate_to_label.items()}




    def _load_and_combine_data(self):
        dataframes = []
        for path in self.paths:
            try:
                df = pd.read_pickle(path)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
        else:
            self.data = pd.DataFrame()

    def _get_rate_list(self):
        rate_list = self.data['mcs'].unique().tolist()
        rate_list.sort()
        mean_rate = np.mean(np.array(rate_list))
        var_rate = np.max(np.array(rate_list)) - np.min(np.array(rate_list))
        rate_list = [(rate-mean_rate) / var_rate for rate in rate_list]
        rate_to_label = {np.float16(rate): label for label, rate in enumerate(rate_list)}
        # label_to_rate = {label: (rate * var_rate + mean_rate) for label, rate in enumerate(rate_list)}

        return rate_list, mean_rate, var_rate, rate_to_label

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # features = row[[f'sinr_{i+1}' for i in range(122)]].values.astype('float32')
        features = row[[f'H_{i+1}' for i in range(122)]]
        features = [np.array(f, dtype='float32').flatten() for f in features]
        features = np.stack(features, axis=0)  # 122, 2
        target = float(row['mcs'])
        target = ((torch.tensor(target, dtype=torch.float32)) - self.mean_rate) / self.var_rate
        label = self.rate_to_label[np.float16(target.item())]
        return torch.tensor(features), target, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)



class InferDataset_noTX(Dataset):
    def __init__(self, infer_paths):

        self.infer_paths = infer_paths
        self._load_and_combine_infer_data()


    def _load_and_combine_infer_data(self):
        dataframes = []
        for path in self.infer_paths:
            try:
                df = pd.read_pickle(path)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if dataframes:
            self.infer_data = pd.concat(dataframes, ignore_index=True)
        else:
            self.infer_data = pd.DataFrame()



    def __getitem__(self, index):
        row = self.infer_data.iloc[index]
        # features = row[[f'sinr_{i+1}' for i in range(122)]].values.astype('float32')
        features = row[[f'H_{i+1}' for i in range(122)]]
        features = [np.array(f, dtype='float32').flatten() for f in features]
        features = np.stack(features, axis=0)  # 122, 2


        return torch.tensor(features), 0, row['dfx_time']

    def __len__(self):
        return len(self.infer_data)

