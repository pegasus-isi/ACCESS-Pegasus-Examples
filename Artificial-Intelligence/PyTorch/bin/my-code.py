#!/usr/bin/env python3

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available.")

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
y = x * 2
print(y)

