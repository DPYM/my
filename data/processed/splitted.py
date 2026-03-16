import os
import pandas as pd

data_path=r'C:/github/project1/data/processed/processed.csv'
data=pd.read_csv(data_path)
print(data.head())