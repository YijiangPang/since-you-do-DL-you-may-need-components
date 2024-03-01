import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import random


class LR_base_DataSet(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    def __getitem__(self, index: int):
        x = self.X_train[index]
        y = self.Y_train[index]
        return x, y
    def __len__(self):
        return len(self.X_train)
    

class LR_base_model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LR_base_model, self).__init__()
        self.clf = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        y = self.clf(x)
        return y
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1: torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


class LogisticRegression:
    gpu = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(gpu)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 32
    lr_init = 1e-2
    epochs = 100
    num_workers = 0

    penalty = "l2"
    l_lambda = 0.1
    def __init__(self, random_state) -> None:
        self.seed = set_random_seed(random_state)

    def fit(self, X_train, Y_train):
        dataset_train = LR_base_DataSet(X_train, Y_train)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size = self.batch_size, num_workers = self.num_workers, drop_last = True, shuffle = True)

        self.model = LR_base_model(in_dim = len(X_train[0]), out_dim = 2).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = len(dataloader_train)*self.epochs)

        self.model.train()
        for i in range(self.epochs):
            for batch_idx, (x, y) in enumerate(dataloader_train):
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                yhat = self.model(x)
                loss = self.cross_entropy_ln_penalty(yhat, y)
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()      
                lr_scheduler.step()

    def predict(self, X_test):
        y_pred = self.predict_proba(X_test)
        return np.round(y_pred[:, 1]).astype(np.int)
    
    def predict_proba(self, X_test):
        dataset_test = LR_base_DataSet(X_test, X_test)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size = self.batch_size, num_workers = self.num_workers, drop_last = False, shuffle = False)
        yhat_all = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, _,) in enumerate(dataloader_test):
                x = x.to(self.device).float()
                yhat = self.model(x)
                yhat = yhat.squeeze().data.cpu().numpy()
                try:
                    yhat[0]
                except:
                    yhat = np.array([yhat.item()])
                yhat_all.append(yhat)
        y_pred = np.concatenate(yhat_all, axis=0)
        return y_pred

    def cross_entropy_ln_penalty(self, yhat, Y_label):
        loss_mean = F.cross_entropy(yhat, Y_label)
        
        if self.penalty == "l2":
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
            l_reg = self.l_lambda * l2_reg
            loss_mean += l_reg
        elif self.penalty == "l1":
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            l_reg = self.l_lambda * l1_reg
            loss_mean += l_reg

        return loss_mean