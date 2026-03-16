import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MultiLabelBinarizer

data_path=r'C:/github/project1/data/raw/'

#读取数据
ratings_df=pd.read_csv(
    data_path+'ratings.dat',
    header=None,
    sep='::',
    engine='python',
    names=['userid','movieid','rating','timestamp']
)

users_df=pd.read_csv(
    data_path+'users.dat',
    header=None,
    sep='::',
    engine='python',
    names=['userid','sex','age','job','zip-code']
)

movies_df=pd.read_csv(
    data_path+'movies.dat',
    header=None,
    sep='::',
    engine='python',
    names=['movieid','moviename','movietype'],
    encoding='ISO-8859-1'
)

#合并
data=ratings_df.merge(movies_df,on='movieid',how='left')
data=data.merge(users_df,on='userid',how='left')

#特征处理
user_encoding=LabelEncoder()
data['userid']=user_encoding.fit_transform(data['userid'])
n_users=len(user_encoding.classes_)

movie_encoding=LabelEncoder()
data['movieid']=movie_encoding.fit_transform(data['movieid'])
n_movies=len(movie_encoding.classes_)

data['sex']=(data['sex']=='M').astype(int)

job_dummis=pd.get_dummies(data['job'],prefix='job')

data['movietype_list']=data['movietype'].apply(lambda x:x.split('|'))
types=set()
for type in data['movietype_list']:
    types.update(type)
types=sorted(list(types))
mlb=MultiLabelBinarizer(classes=types)
types_encode=mlb.fit_transform(data['movietype_list'])
types_df=pd.DataFrame(types_encode,columns=[f'type_{t}' for t in types])

data['timestamp']=pd.to_datetime(data['timestamp'],unit='s')
data['hour']=data['timestamp'].dt.hour
data['day_of_week']=data['timestamp'].dt.dayofweek
data['month']=data['timestamp'].dt.month

#合并特征
features_data=pd.concat([
    data[['userid','movieid','sex','age','hour','day_of_week','month','rating']],
    job_dummis,
    types_df],
    axis=1
)
print(features_data.head())
print(f'数量:{features_data.shape[1]}')

features_data.to_csv(r'C:\github\project1\data\processed\processed.csv',index=False)