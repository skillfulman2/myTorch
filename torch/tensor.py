import torch
import numpy as np


# Tensors are specialized 

# Can be created directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


# tensors can be created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)






