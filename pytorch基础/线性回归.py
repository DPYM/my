import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

x=torch.randn(100,2)
true_w=torch.tensor([2.0,3.0])
true_b=4.0
y=x@true_w+true_b+torch.randn(100)*0.05

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(2,1)
    
    def forward(self,x):
        return self.linear(x)
    
model=LinearRegressionModel()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.05)

rounds=1000
for i in range(rounds):
    model.train
    
    predictions=model.forward(x)
    loss=criterion(predictions.squeeze(),y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%100==0:
        print(f'第{i+1}次的损失为{loss:.4f},斜率为{model.linear.weight.mean().item():.3f},截距为{model.linear.bias.item():.3f}')

with torch.no_grad():
    predictions=model(x)

plt.figure(figsize=(10,6))
plt.scatter(x[:,0],y,color='red',s=10)
plt.scatter(x[:,0],predictions,color='blue',s=10)
plt.show()