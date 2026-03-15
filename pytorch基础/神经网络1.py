import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda')

#定义一个模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x)) #定义激活函数，有ReLU,SoftMax,Sigmoid,Tanh
        x = self.fc2(x)
        return x
    
model = SimpleNN() 
model.to(device)

X = torch.randn(100,2)
Y = torch.randn(100,1)

X = X.to(device)
Y = Y.to(device)

criterion = nn.MSELoss()  #定义损失函数，有MSE,交叉熵
optimizer = optim.Adam(model.parameters(),lr=0.005)  #定义优化器，有Adam,SGD,RMSprop

for epoch in range(100):
    target = model(X)  #预测
    loss = criterion(target,Y)  #计算损失
    optimizer.zero_grad()  #清除梯度
    loss.backward()  #计算梯度
    optimizer.step()   #更新参数模型

    if (epoch+1)%10==0:
        print(f'学习{epoch+1}次后，此时的损失是{loss}')