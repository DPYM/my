import pandas as pd
import os

data_path=r'C:/github/project1/data/raw'

def data_loader(data_path):
    data_name='ratings.dat'
    full_name=os.path.join(data_path,data_name)
    ratings=pd.read_csv(
        full_name,
        header=None,
        sep='::',
        engine='python',
        names=['Userid','Movieid','rating','Timestamp']
)
    n_user=ratings['Userid'].unique()
    n_movie=ratings['Movieid'].unique()
    n_interactions=len(ratings)
    
    ratings=ratings.sort_values(['Userid','Timestamp'])
    users_squeeze=ratings.groupby('Userid')['Movieid'].apply(list)

    print(users_squeeze)

data_loader(data_path)