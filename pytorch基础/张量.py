import torch
import numpy as np

dtype = torch.float
device = torch.device('cuda')


'''x = torch.arange(0,10,2)
y = torch.linspace(0,1,9)
eye = torch.eye(4)
print(x)
print(y)
print(eye)
'''
'''
tensor_d = torch.randn(1,4,device=device)*10
print(tensor_d)

tensor_2d = torch.stack([tensor_d,tensor_d-3,tensor_d+5])
print(tensor_2d)
print(tensor_2d.shape)
print(tensor_2d.size())
print(tensor_2d.dtype)
print(tensor_2d.device)
print(tensor_2d.is_contiguous())

mask = tensor_2d>5
print(mask)
filled_tensor = tensor_2d[tensor_2d>5].t()
print(filled_tensor)'''

'''
tensor_3d = torch.stack([tensor_2d,tensor_2d+3,tensor_2d-5])
print(tensor_3d)
print(tensor_3d.shape)

tensor_4d = torch.stack([tensor_3d,tensor_3d-10,tensor_3d+20])
print(tensor_4d)
print(tensor_4d.shape)
'''

'''
np_array = np.array([[1,2,3],[4,5,6]])
print(np_array)
x = torch.from_numpy(np_array)
print(x)
np_array[0,0]=7
print(np_array)
print(x)
'''
tensor = torch.tensor([[1,2,3],[4,5,6]])
print(tensor)
np_from_tensor_independent = tensor.clone().numpy()
print(np_from_tensor_independent)
tensor[0,0]=7
print(tensor)
print(np_from_tensor_independent)