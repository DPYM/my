import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader


data_path=r'C:/github/project1/data/splited_data/'
train_df=pd.read_csv(data_path+'train.csv')
print(train_df.head())

'''#定义封闭接口
class User_Movie_Dataset(Dataset):
    def __init__(self,train_df):
        self.user_ids=torch.tensor(train_df['userid'].values)
        self.momvie_ids=torch.tensor(train_df['movieid'].values)
        self.labels=(train_df['rating']>=4).astype(int).values

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self,idx):
        return self.user_ids[idx],self.momvie_ids[idx],self.labels[idx]
    
train_df=User_Movie_Dataset(train_data)'''