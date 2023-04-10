import numpy as np
from gym import Env
from .contextual_unlock import ContextualUnlock
from typing import Optional

class ContextualUnlock1DInMBRL(Env):

    def __init__(self, context=np.array([ContextualUnlock.ROOM_SIZE//2])):
        self.env = ContextualUnlock(test_mode='IID', context=np.concatenate((context, [ContextualUnlock.ROOM_SIZE//2])))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = np.concatenate((np.round(context), [0])).astype(int)

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = self.env.step_count >= self.env.max_steps
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        self.env.render(mode)