import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)

model = DecisionTreeClassifier(max_depth=7)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)

print(accuracy,MSE,RMSE)
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()