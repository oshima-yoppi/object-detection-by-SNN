import time
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


matrix_size = 30*500

x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)

# print("************ CPU SPEED ***************")
# start = time.time()
# result = torch.matmul(x,y)
# print(time.time() - start)
# print("verify device:", result.device)

x_gpu = x.to(device)
y_gpu = y.to(device)
torch.cuda.synchronize()

for i in range(3):
    print("************ GPU SPEED ***************")
    start = time.time()
    result_gpu = torch.matmul(x_gpu,y_gpu)
    torch.cuda.synchronize()
    print(time.time() - start)
    print("verify device:", result_gpu.device)