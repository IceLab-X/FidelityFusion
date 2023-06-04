import torch
import numpy as np

model = torch.jit.load('module.pt')
args = np.load('args.npy', allow_pickle=True)
kwargs = np.load('kwargs.npy', allow_pickle=True).item()

torch_args = [torch.tensor(_v) if hasattr(_v, 'shape') else _v for _v in args]
torch_kwargs = {}
for _key, _v in kwargs.items():
    torch_kwargs[_key] = torch.tensor(_v) if hasattr(_v, 'shape') else _v

model(*torch_args, **torch_kwargs)

