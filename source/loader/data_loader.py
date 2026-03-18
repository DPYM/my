import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer

data_path=r'C:/github/project1/data/origin_data/'

#读取数据
ratings_df=pd.read_csv(
    data_path+'ratings.dat',
    header=None,
    sep='::',
    engine='python',
    names=['userid','movieid','rating','timestamp']
)
#print(ratings_df.head())
#print(ratings_df.info())
#print(ratings_df.isnull().sum())

users_df=pd.read_csv(
    data_path+'users.dat',
    header=None,
    sep='::',
    engine='python',
    names=['userid','sex','age','job','zip-code']
)
#print(users_df.head())
#print(users_df.info())
#print(users_df.isnull().sum())

movies_df=pd.read_csv(
    data_path+'movies.dat',
    header=None,
    sep='::',
    engine='python',
    names=['movieid','moviename','movietype'],
    encoding='ISO-8859-1'
)
#print(movies_df.head())
#print(movies_df.info())
#print(movies_df.isnull().sum())

#合并
data=ratings_df.merge(movies_df,on='movieid',how='left')
data=data.merge(users_df,on='userid',how='left')

#特征处理
#用户id编码，标签编码
user_encoding=LabelEncoder()
data['userid']=user_encoding.fit_transform(data['userid'])
n_users=len(user_encoding.classes_)

#电影id编码，标签编码
movie_encoding=LabelEncoder()
data['movieid']=movie_encoding.fit_transform(data['movieid'])
n_movies=len(movie_encoding.classes_)

#性别采用硬编码
data['sex']=(data['sex']=='M').astype(int)

#工作采用独热编码
job_dummis=pd.get_dummies(data['job'],prefix='job')

#对于电影类别采用多热编码
data['movietype_list']=data['movietype'].apply(lambda x:x.split('|'))
types=set()
for type in data['movietype_list']:
    types.update(type)
types=sorted(list(types))
mlb=MultiLabelBinarizer(classes=types)
types_encode=mlb.fit_transform(data['movietype_list'])
types_df=pd.DataFrame(types_encode,columns=[f'type_{t}' for t in types])

#日期转换
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

#划分训练集，验证集，测试集，比例为7:1:2
train_data=[]
val_data=[]
test_data=[]

#创建用户历史行为列表
user_history={}

for user_id,user_val in features_data.groupby('userid'):
    #按时间排序
    user_val=user_val.sort_values('timestamp')

    n=len(user_val)
    #把行为少于三次的直接放训练集里面
    if n<3:
        train_data.append(user_val)
        continue

    movie_idx=user_val['movieid'].tolist()
    user_history[user_id]=movie_idx[:int(n*0.7)]

    train_data.append(user_val.iloc[:int(n*0.7)])
    val_data.append(user_val.iloc[int(n*0.7):int(n*0.8)])
    test_data.append(user_val.iloc[int(n*0.8):])

#合并数据
train_df=pd.concat(train_data,ignore_index=True)
val_df=pd.concat(val_data,ignore_index=True)
test_df=pd.concat(test_data,ignore_index=True)

train_df=train_df.drop(columns='timestamp')
val_df=val_df.drop(columns='timestamp')
test_df=test_df.drop(columns='timestamp')

#保存文件
train_df.to_csv(r'C:/github/project1/data/splited_data/train.csv',index=False)
val_df.to_csv(r'C:/github/project1/data/splited_data/val.csv',index=False)
test_df.to_csv(r'C:/github/project1/data/splited_data/test.csv',index=False)

with open(r'C:/github/project1/data/processed_data/user_history.pkl','wb') as f:
    pickle.dump(user_history,f)