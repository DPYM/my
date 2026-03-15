import torch.nn as nn
import torch 
import torchvision.datasets as datasets
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from PIL import Image

dtype=torch.float

'''class MyDataset(nn.Module):
    def __init__(self,x_data,y_data):
        self.x=x_data
        self.y=y_data
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, key):
        x=torch.tensor(self.x[key])
        y=torch.tensor(self.y[key])
        return x,y
    
x_data=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
y_data=[1,0,1,0]

dataset=MyDataset(x_data,y_data)

dataloader=DataLoader(dataset,batch_size=2,shuffle=True)

for epoch in range(1):
    for idx,(inputs,labels) in enumerate(dataloader):
        print(f'batch{idx}')
        print(f'input{inputs}')
        print(f'label{labels}')'''

'''transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48,0.486,0.49],std=[0.22,0.21,0.215])
])

image=Image.open('pytorch基础/1.jpg')
image_tensor=transform(image)
print(image_tensor.shape)'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset=datasets.MNIST(root='/data',train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root='/data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)

for inputs,labels in train_loader:
    print(inputs.shape)
    print(labels.shape)