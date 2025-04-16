import torch 

ACT2FN = {
    'relu': torch.nn.functional.relu,
    'gelu': torch.nn.functional.gelu,
    'silu': torch.nn.functional.silu,
    'swish': torch.nn.functional.silu,
}