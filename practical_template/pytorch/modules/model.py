# -*- coding: utf-8 -*-
#%%
import torch
import torch.nn as nn

import modules.dataset as dataset

class V1(nn.Module):
    def __init__(self, in_size, out_size):
        super(V1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # input layer
        self.in_layer = nn.Linear(in_size, 50)
        
        # sequential layer
        self.net = nn.Sequential(
                nn.Linear(50,50),
                nn.ReLU(),
                nn.Linear(50,50),
                nn.ReLU(),
                nn.Linear(50,50),
                nn.ReLU()
                )
        
        
        # output layer
        self.out_layer = nn.Sequential(
                nn.Linear(50, out_size),
                nn.Sigmoid()
                )
        
    def forward(self, X):
        X = self.in_layer(X)
        X = self.net(X)
        X = self.out_layer(X)
        return X

 #%%   
if __name__ == "__main__":
    config = {
        "X" : ["gender","my_pref","my_rating","partner_rating","interest","guess"],
        "Y" : ["partner"]
        }
    
    ds = dataset.SpeedDating_Dataset(config)
    X,Y = ds[0]
    print(X)
    print(Y)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = V1(int(X.shape[0]), int(Y.shape[0])).to(device)
    Y_ = model(X)
    print(Y_)
    