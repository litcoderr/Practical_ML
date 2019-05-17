import torch
import torch.utils.data as data
import pandas as pd
import os
#%%
class SpeedDating_Dataset(data.Dataset):
    def __init__(self, config, mode="train", test_p=0.1):
        super(SpeedDating_Dataset).__init__();
        #  Constants
        self.dataset_dir = "/Users/jameschee/Desktop/ml_tutorial/practical_template/pytorch/data/speed_dating.csv"
        self.config_ = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.test_p = test_p
        self.mode = mode
        
        # load data
        raw_data = pd.read_csv(self.dataset_dir)
        raw_data = raw_data.sample(frac=1).reset_index(drop=True)
        self.data = self.preprocess(raw_data)
        
        
    def __len__(self):
        length = self.data[0].shape[0]
        return length


    def __getitem__(self, index):
        x = torch.Tensor(self.data[0].iloc[index].values).to(self.device)
        y = torch.Tensor(self.data[1].iloc[index].values).to(self.device)
        return x,y
    
    
    def preprocess(self, data):
        # 1. get item we want
        self.item = self.get_valid_item(data)
        
        # 2. filter out null datasets
        data = data.loc[(data["has_null"]==0)&(data["decision"]==1)]
        
        # 3. get wanted item
        X = data.loc[:,self.to_list(self.config_["X"])]
        Y = data.loc[:,self.to_list(self.config_["Y"])]
        
        # 4. rescale to 0 to 1
        X = self.rescale(X).reset_index(drop=True)
        Y = self.rescale(Y).reset_index(drop=True)
        
        if self.mode == "train":
            boundary = int(X.shape[0]*(1-self.test_p))
            X = X.loc[0:boundary].reset_index(drop=True)
            Y = Y.loc[0:boundary].reset_index(drop=True)
        elif self.mode == "test":
            boundary = int(X.shape[0]*(1-self.test_p))+1
            X = X.loc[boundary:].reset_index(drop=True)
            Y = Y.loc[boundary:].reset_index(drop=True)
        
        return X,Y
    
    
    def rescale(self, data):
        try:
            data.loc[data.gender != "female", "gender"] = 1
            data.loc[data.gender == "female", "gender"] = 0
        except:
            pass
        
        rescale_val = {
                "my_pref" : 100.0,
                "my_rating" : 10.0,
                "partner_rating" : 10.0,
                "interest" : 1.0,
                "guess" : 10.0,
                "partner" : 1.0
                }
        
        for key in rescale_val.keys():
            for col in self.to_list([key]):
                try:
                    data.loc[:,col] = data.loc[:,col] / rescale_val[key]
                except:
                    pass
                
        return data
        
    
    def get_valid_item(self, data):
        column = list(data.columns.values)
        result = {}
        result["gender"] = ["gender"]
        # my preference
        result["my_pref"] = ([item for item in column if "_important" in item and "d_" not in item])
        # rate of myself
        result["my_rating"] = column[column.index("attractive"):column.index("ambition")+1]
        # rate of partner
        result["partner_rating"] = [item for item in column if "_partner" in item and "d_" not in item]
        # interest correlation
        result["interest"] = ["interests_correlate"]
        # guess likeness
        result["guess"] = ["guess_prob_liked"]
        # partner's choice
        result["partner"] = ["decision_o"]
        return result
    
    
    def to_list(self, target):
        result = []
        if target != []:
            for key in target:
                result += self.item[key]
        else:
            for key in self.item.keys():
                result += self.item[key]
            
        return result
#%%
    class StockDataset(data.Dataset):
        def __init__(self):
            super(StockDataset, self).__init__()
            pass
        
        def __len__(self):
            pass
        
        def __getitem__(self, index):
            pass
    
#%%
if __name__ == "__main__":
    config = {
                "X" : ["gender","my_pref","my_rating","partner_rating","interest","guess"],
                "Y" : ["partner"]
                }
    dataset = SpeedDating_Dataset(config,mode="train")
    print(len(dataset))