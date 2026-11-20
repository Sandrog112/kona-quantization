import torch

if torch.backends.mps.is_available():
    x = torch.ones(5, device="mps")
    print("MPS is available:", x * 2)
else:
    print("MPS is not available")
