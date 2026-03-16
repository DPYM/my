import pandas as pd
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer

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
#用户id编码，使用
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
    data[['userid','movieid','sex','age','hour','day_of_week','month','rating','timestamp']],
    job_dummis,
    types_df],
    axis=1
)

train_data=[]
val_data=[]
test_data=[]

#划分训练集，验证集，测试集，比例为0.7:0.1:0.2
for user_id,user_val in features_data.groupby('userid'):
    user_val.sort_values('timestamp')

    n=len(user_val)
    #把行为少于三次的直接放训练集里面
    if n<3:
        train_data.append(user_val)
        continue
    
    train_data.append(user_val.iloc[:int(n*0.7)])
    val_data.append(user_val.iloc[int(n*0.7):int(n*0.8)])
    test_data.append(user_val.iloc[int(n*0.8):])

train_df=pd.concat(train_data,ignore_index=True)
val_df=pd.concat(val_data,ignore_index=True)
test_df=pd.concat(test_data,ignore_index=True)

train_df.to_csv(r'C:/github/project1/data/split/train.csv',index=False)
val_df.to_csv(r'C:/github/project1/data/split/val.csv',index=False)
test_df.to_csv(r'C:/github/project1/data/split/test.csv',index=False)