import torch


def one_hot_code(values, value_dim, device):
    code = torch.zeros((values.shape[0], value_dim), dtype=torch.float32, device=device)
    code = code.scatter(1, values, 1.0)
    return code
