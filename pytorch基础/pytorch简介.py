#pytorch特点：动态计算图，自动微分，张量计算，丰富API，多语言支持
import torch
dtype = torch.float
device = torch.device('cuda')

'''a = torch.randn(2,3,device=device,dtype=dtype)
b = torch.randn(2,3,device=device,dtype=dtype)

print(a)
print(b)
print(a*b)
print(a.sum())
print(a[:,1])
print(a.max())

c = torch.zeros(3,2)
d = torch.ones(3,2)
e = torch.randn(3,2)
print(c)
print(d)
print(e)'''

import numpy as np

'''f = np.random.randn(3,2)
t_from_np1 = torch.tensor(f)
t_from_np2 = torch.from_numpy(f)

print(f)
print(t_from_np1)
print(t_from_np2)'''

'''tensor_requires_grad = torch.tensor(
    [1.0],
    requires_grad=True
)
tensor_result = tensor_requires_grad*2
tensor_result.backward()
print(tensor_requires_grad.grad)'''

x = torch.randn(2,2,requires_grad=True)
print(x)
y = x*2
z = y.sum()**2+5
z.sum().backward()

print(x.grad)
print(z.mean())