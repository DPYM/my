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