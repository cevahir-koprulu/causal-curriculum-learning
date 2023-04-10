import torch

def unlock(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    env_size = next_obs.shape[1]-2//3
    door_pos = next_obs[:, 2*env_size:3*env_size]
    return (torch.sum(door_pos, axis=1) == 0).reshape(-1,1)