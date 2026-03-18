import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)
accuracies = []
for k in range(1,21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracies.append(accuracy)



x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
y_min,y_max = x[:,1].min()-1,x[:,1].max()+1

xx,yy = np.meshgrid(
    np.arange(x_min,x_max,0.01),
    np.arange(y_min,y_max,0.01)
)

z = model.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.contourf(
    xx,
    yy,
    z,
    cmap='coolwarm'
)
plt.scatter(x[:,0],x[:,1],s=10,edgecolor='black',color='yellow')

plt.subplot(2,1,2)
plt.plot(range(1,21),accuracies,lw=2,marker='o')
plt.show()