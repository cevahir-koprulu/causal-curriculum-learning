'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-03-10 12:28:43
LastEditTime: 2022-07-28 17:13:25
Description: 
'''

from deep_sprl.environments.unlock.env.roomgrid import RoomGrid
from typing import Optional
import numpy as np

class ContextualUnlock(RoomGrid):
    """
    Unlock a door
    """
    ROOM_SIZE = 10 # 12

    def __init__(self, test_mode='IID', stage="train", max_steps=100, seed=None, context=np.array([ROOM_SIZE//2, ROOM_SIZE//2])):
        assert test_mode in ['IID', 'OOD-E', 'OOD-S']
        self.test_mode = test_mode
        self.stage = stage
        self.room_size = self.ROOM_SIZE
        self.context = context
        super().__init__(num_rows=1, num_cols=1, room_size=self.room_size, max_steps=max_steps, seed=seed, stage=stage)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height, self.context)

        # place an agent in the room (0, 0)
        self.place_agent(0, 0)

        # extrpolation setting
        if self.test_mode == 'OOD-E':
            # in training stage, we only have one door
            if self.stage == 'train':
                door_idx = np.random.choice([0, 1])
                door, door_pos = self.add_door(0, 0, door_idx=door_idx, color=None, locked=True)
            # in testing stage, there are two doors
            else:
                door_idx_1 = 0
                door_idx_2 = 1
                door, door_pos = self.add_door(0, 0, door_idx=door_idx_1, color=None, locked=True)
                door_2, door_pos_2 = self.add_door(0, 0, door_idx=door_idx_2, color=door.color, locked=True)
                assert door.color == door_2.color
        else:
            door_idx = 0
            door, door_pos = self.add_door(0, 0, door_idx=door_idx, color=None, locked=True)

        # spurious setting
        if self.test_mode == 'OOD-S':
            # in training stage, the key and door should be put in the same row
            if self.stage == 'train':
                fixed_row = door_pos[1]
            else:
                # in testing stage, the key and door should be in different rows
                candidate = list(range(1, self.room_size-1))
                candidate.remove(door_pos[1])
                fixed_row = np.random.choice(candidate)
        else:
            fixed_row = None

        # add one key
        self.add_object(0, 0, 'key', door.color, fixed_row)
        self.mission = "unlock the door"

    def reset(self, seed: Optional[int] = None):
        obs = super().reset(seed=seed)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["success"] = False
        if reward == 1:
            info["success"] = True
        return obs, reward, done, info
