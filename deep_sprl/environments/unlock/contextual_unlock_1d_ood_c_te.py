import numpy as np
from gym import Env
from .contextual_unlock import ContextualUnlock

class ContextualUnlock1DOoDCTe(Env):

    def __init__(self, context=np.array([ContextualUnlock.ROOM_SIZE//2, ContextualUnlock.ROOM_SIZE//2])):
        self.env = ContextualUnlock(test_mode='OOD-C', stage="test", max_steps=150, context=context)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = np.round(context).astype(int)

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)