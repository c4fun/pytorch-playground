import torch
import numpy as np

# initializing a tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# initializing a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# initializing another tensor from a tensor
x_zero = torch.zeros_like(x_data)
print(f'Zeros Tensor: \n {x_zero} \n')

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# initializing a tensor with random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


## Attributes of a Tensor
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

## Operations on Tensors
# explicitly move tensors to the GPU using .to method (after checking for GPU availability)
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])  # ellipsis to indicate that the rest of the dimensions are implied
tensor[:, 1] = 0
print(tensor)

# joining tensors  列方向に結合。注意这个是一个二维的tensor，所以dim取值范围在[-2,1]
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


## Bridge with NumPy

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()  # tensor to numpy array
print(f"n: {n}")

# 注意这里是引用
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

