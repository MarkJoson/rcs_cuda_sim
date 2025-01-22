import cudasim_pycore as csc
import torch as th
a = th.Tensor([1,2,3,4])
b = th.Tensor([5,6,7,8])

# csc.tensor_add(a, b)

a = csc.GTensor.zeros([16,32], csc.kFloat32, csc.kCPU)
b = a.getTorchTensor()

b += 55

a.print()
# print(b)