from deep_sprl.environments.unlock.env.termination_fn import unlock as unlock_termination
import torch

def unlock(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    return (unlock_termination(act, next_obs)).float().reshape(-1,1)