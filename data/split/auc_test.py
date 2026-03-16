import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train=pd.read_csv(r'C:/github/project1/data/split/train.csv')
val=pd.read_csv(r'C:/github/project1/data/split/val.csv')
test=pd.read_csv(r'C:/github/project1/data/split/test.csv')

#设置特征和标签
x_train=train.drop(columns=['rating'])
y_train=(train['rating']>=4).astype(int)

x_val=val.drop(columns=['rating'])
y_val=(val['rating']>=4).astype(int)

x_test=test.drop(columns=['rating'])
y_test=(test['rating']>=4).astype(int)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

print(f"训练集的auc:{roc_auc_score(y_train,model.predict_proba(x_train)[:,1]):.4f}")
print(f"验证集的auc:{roc_auc_score(y_val,model.predict_proba(x_val)[:,1]):.4f}")
print(f"训练集的auc:{roc_auc_score(y_test,model.predict_proba(x_test)[:,1]):.4f}")

'''
基准线：
训练集的auc:0.6006
验证集的auc:0.5891
训练集的auc:0.5966
这里使用了逻辑回归的简单模型对数据集进行了初步训练，但训练出的AUC分数在0.6左右，并没有达到预想中的
大于0.65，判断为模型过于简单，也有可能是当前的数据过于原始还没有进行加工，接下来会对数据进行加工处
理，并使用更加强大的模型对数据进行进一步训练，观察能否能对训练效果进行进一步的优化
'''