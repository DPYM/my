import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
#特征工程：只选择全两列进行分析
iris = load_iris()
x = iris.data[:,:2]
y = (iris.target!=0)*1
#划分测试集，训练集
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)
#选择模型
model = LogisticRegression()
#绘制学习曲线
train_sizes,train_scores,val_scores = learning_curve(
    model,
    x_train,
    y_train,
    cv=5
)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
matrix = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)

x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
y_min,y_max = x[:,1].min()-1,x[:,1].max()+1

xx,yy = np.meshgrid(
    np.arange(x_min,x_max,0.01),
    np.arange(y_min,y_max,0.01)
)

z = model.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)


print(f'准确率:{accuracy}\n 混淆矩阵:\n{matrix}\n 分类报告:\n{report}')
plt.figure(figsize=(10,10))
#画学习曲线
plt.subplot(2,1,1)
plt.plot(train_sizes,train_scores.mean(axis=1),'r--',label='训练集')
plt.plot(train_sizes,val_scores.mean(axis=1),'b--',label='测试集')

#画决策边界
plt.subplot(2,1,2)
plt.contourf(xx,yy,z,cmap='coolwarm')
plt.scatter(x[:,0],x[:,1],s=10,c=y,edgecolors='black')
plt.show()