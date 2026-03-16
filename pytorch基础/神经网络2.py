import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

dtype = torch.float

'''
x_in,x_hide,x_out,hatch_size=10,5,1,10

X = torch.randn(hatch_size,x_in)
Y = torch.tensor(
    [[1.0],[0.0],[0.0],[1.0],[1.0],[0.0],[0.0],[1.0],[0.0],[1.0]]
)

model = nn.Sequential(
    nn.Linear(x_in,x_hide),
    nn.ReLU(),
    nn.Linear(x_hide,x_out),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.005)
losses = []
epoch = 100

for i in range(epoch):
    y_pred = model(X)
    loss = criterion(Y,y_pred)
    losses.append(loss.item())
    if (i+1)%10==0:
        print(f'迭代次数为{i+1}时，损失为{loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure(figsize=(10,6))
plt.plot(range(epoch),losses,label='loss')
plt.legend()
plt.grid()
plt.show()

y_final_pred = model(X).detach().numpy()
y_actual = Y.numpy()

plt.figure(figsize=(10,6))
plt.plot(range(1,hatch_size+1),y_actual,color='red',lw=2)
plt.plot(range(1,hatch_size+1),y_final_pred,color='blue',lw=2)
plt.show()'''

s_samples = 500
data = torch.randn(s_samples,2)
label = (data[:,0]**2+data[:,1]**2<1).float().unsqueeze(1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.005)
epoch = 100

for i in range(epoch):
    target = model(data)
    loss = criterion(data,target)
    if (i+1)%10==0:
        print(f'迭代次数为{i+1}时，损失为{loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_decision_boundary(model,data):
    x_min,x_max = data[:,0].min()-1,data[:,0].max()+1
    y_min,y_max = data[:,1].min()-1,data[:,1].max()+1

    xx,yy = torch.meshgrid(
        torch.arange(x_min,x_max,0.01),
        torch.arange(y_min,y_max,0.01)
    )

    grid = torch.cat([xx.reshape(-1,1),yy.reshape(-1,1)],dim=1)
    predictions = model(grid).detach().numpy().reshape(xx.shape)

    plt.figure(figsize=(10,6))
    plt.contourf(
        xx.numpy(),
        yy.numpy(),
        predictions,
        cmap='coolwarm',
    )
    plt.scatter(data[:,0],data[:,1],s=10,edgecolor='black',color='yellow')
    plt.show()
plot_decision_boundary(model,data)