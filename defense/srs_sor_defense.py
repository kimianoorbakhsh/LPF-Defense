import numpy as np
import torch

def duplicate_randomly(pc, size):  
    """ Make up for the point loss due to conflictions in the projection process. """
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))
    
def SRS(pc, drop_num=500):
    """ SRS Defense: Randomly Dropping Points. Taken from: https://github.com/Wuziyi616/IF-Defense/tree/main/baselines/defense/drop_points """
    K = pc.shape[0]
    idx = np.random.choice(K, K - drop_num, replace=False)
    return pc[idx]
    
def SOR(x, k=2, alpha=1.1):
    """ SOR Defense: Statistical Outlier Removal """
    x = torch.from_numpy(x.reshape((1, ) + x.shape)).to("cpu")  # so that it seems like a batch of point clouds
    pc = x.clone().detach().double()
    B, K = pc.shape[:2]
    pc = pc.transpose(2, 1)  # [B, 3, K]
    inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
    xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
    dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
    assert dist.min().item() >= -1e-6
    # the min is self so we take top (k + 1)
    neg_value, _ = (-dist).topk(k=k + 1, dim=-1)  # [B, K, k + 1]
    value = -(neg_value[..., 1:])  # [B, K, k]
    value = torch.mean(value, dim=-1)  # [B, K]
    mean = torch.mean(value, dim=-1)  # [B]
    std = torch.std(value, dim=-1)  # [B]
    threshold = mean + alpha * std  # [B]
    bool_mask = (value <= threshold[:, None])  # [B, K]
    sel_pc = [x[i][bool_mask[i]] for i in range(B)]
    return sel_pc[0].to("cpu").numpy()
    
