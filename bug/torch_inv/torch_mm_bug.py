# From: https://github.com/pytorch/pytorch/issues/56747import torch

import torch

def matmul_test(mat_a, mat_b, dtype, device):
    print("matmul_test -- ", dtype, device)
    a = torch.tensor(mat_a, dtype=dtype, device=device,)
    b = torch.tensor(mat_b, dtype=dtype, device=device,)
    c = torch.matmul(b, a)
    print("ba:\n", c)

def main():
    a, b = [[1, 2]], [[3], [4]]
    print("a:", a, "\nb:", b)
    matmul_test(a, b, torch.float16, torch.device("cpu"))
    matmul_test(a, b, torch.float16, torch.device("cuda"))
    matmul_test(a, b, torch.float32, torch.device("cuda"))
    matmul_test(a, b, torch.float64, torch.device("cuda"))
    matmul_test(a, b, torch.float16, torch.device("cuda"))

assert __name__ == "__main__"
main()
