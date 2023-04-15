from copy import deepcopy
import math
import numpy as np
from gym.utils import seeding
from gym import spaces
from typing import Optional
from deep_sprl.environments.unlock.env.rendering import fill_coords, point_in_rect, point_in_line, point_in_circle, downsample, highlight_img, point_in_triangle, rotate_fn


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100])
}
COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        # originally, the door can be overlaped if it is open, but we dont allow it since there is noly one room
        #return self.is_open
        return False

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    # this is different from the original Minigrid env
    def can_overlap(self):
        return True

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Grid:
    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height, f"i,j={i,j} for v={v}"
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_circle(0.5, 0.5, 0.3)

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos=None, agent_dir=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(cell, agent_dir=agent_dir if agent_here else None, highlight=highlight_mask[i, j], tile_size=tile_size)

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class MiniGridEnv:
    """
    2D grid world game environment.
    All baisc functions are implemented here.
    """

    def __init__(self, width=None, height=None, max_steps=100, seed=1337, stage="train"):
        # stage
        self.stage = stage

        # action dim
        self.move_dim = 4 
        self.pick_key_dim = 2 # [pick, not pick]
        self.open_door_dim = 2 # [open, not open]
        self.action_dim = self.move_dim + self.pick_key_dim + self.open_door_dim
        self.action_space = spaces.Box(np.zeros((self.action_dim,)), np.ones((self.action_dim,)))
        # [down, up, left, right]
        self.move_mapping = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        # the original grid is too complex to use, we directly save the state in an array
        # [agent, key, door]
        # self.state = np.zeros((width, height, 3)) # OLD STATE
        self.state = np.zeros((width+height, 3)) # NEW STATE

        # this should also be one-hot for each key
        self.max_key_num = 2
        self.have_key = np.zeros((self.max_key_num,))

        # state dim
        # self.ego_app_dim = width*height # OLD STATE
        # self.key_app_dim = width*height # OLD STATE
        # self.door_app_dim = width*height # OLD STATE
        self.ego_app_dim = width+height # NEW STATE
        self.key_app_dim = width+height # NEW STATE
        self.door_app_dim = width+height # NEW STATE
        self.state_dim = self.ego_app_dim + self.key_app_dim + self.door_app_dim + self.max_key_num
        self.observation_space = spaces.Box(np.zeros((self.state_dim,)), np.ones((self.state_dim,)))

        # environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # initiliazation
        self.window = None
        self.reset(seed=seed)

    def random_action(self):
        move = np.random.randint(0, self.move_dim)
        pick_key = np.random.randint(0, self.pick_key_dim)
        open_door = np.random.randint(0, self.open_door_dim)
        action = np.zeros((self.action_dim))
        action[move] = 1.0
        action[self.move_dim+pick_key] = 1.0
        action[self.move_dim+self.pick_key_dim+open_door] = 1.0
        return action

    def reset(self, seed: Optional[int] = None):
        self.seed(seed=seed)
        # print("=====================================")
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)
        # self.state = np.zeros((self.width, self.height, 3)) # OLD STATE
        self.state = np.zeros((self.width+self.height, 3)) # NEW STATE
        self.have_key = np.zeros((self.max_key_num,))
        self.have_key[0] = 1.0

        # save the position of door and key
        for idx, g_i in enumerate(self.grid.grid):
            if g_i and g_i.type == 'door':
                # self.state[idx // self.height, idx % self.width, 2] = 1.0 # OLD STATE
                self.state[idx % self.width, 2] = 1.0 # NEW STATE
                self.state[self.width + idx // self.height, 2] = 1.0 # NEW STATE
            elif g_i and g_i.type == 'key':
                # self.state[idx // self.height, idx % self.width, 1] = 1.0 # OLD STATE
                self.state[idx % self.width, 1] = 1.0 # NEW STATE
                self.state[self.width + idx // self.height, 1] = 1.0 # NEW STATE

        # save the position of agent
        # self.state[self.agent_pos[1], self.agent_pos[0], 0] = 1.0 # OLD STATE
        self.state[self.agent_pos[0], 0] = 1.0 # NEW STATE
        self.state[self.width + self.agent_pos[1], 0] = 1.0 # NEW STATE

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _gen_grid(self, width, height):
        raise NotImplementedError("_gen_grid needs to be implemented by each environment")

    def _rand_int(self, low, high):
        return self.np_random.randint(low, high)

    def _rand_bool(self):
        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_color(self):
        return self._rand_elem(COLOR_NAMES)

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf, fixed_row=None):
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        # random generate
        num_tries = 0
        while True:
            # This is to handle with rare cases where rejection sampling``
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            if obj is None:
                col = self._rand_int(top[0], min(top[0] + size[0]/4, self.grid.width/4)+1)
                row = self._rand_int(top[1], min(top[1] + size[1]/4, self.grid.height/4)+1)
            elif obj.type == "key":
                col = self._rand_int(top[0], min(top[0] + size[0]/2, self.grid.width/2)+1)
                row = self._rand_int(top[1], min(top[1] + size[1]/2, self.grid.height/2)+1)
            else:
                col = self._rand_int(top[0], min(top[0] + size[0], self.grid.width))
                row = self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
                
            if fixed_row is not None:
                row = fixed_row
            pos = np.array((col, row))

            # Don't place the object on top of another object
            # Don't place the object where the agent is
            # Check if there is a filtering criterion
            #if self.grid.get(*pos) is not None or np.array_equal(pos, self.agent_pos) or (reject_fn and reject_fn(self, pos)): 
            if self.grid.get(*pos) is not None or (reject_fn and reject_fn(self, pos)): 
                continue
            break

        self.grid.set(*pos, obj)
        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries, fixed_row=None)
        self.agent_pos = pos
        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)
        return pos

    def step(self, action):
        self.step_count += 1
        # process action - [M(4), P(2), O(2)]
        move = action[0:self.move_dim]
        pick_key = action[self.move_dim:self.move_dim+self.pick_key_dim]
        open_door = action[self.move_dim+self.pick_key_dim:]
        # print(f"Actions: move={move} || pick_key={pick_key} || open_door={open_door}")
        assert move.shape[0] == self.move_dim
        assert pick_key.shape[0] == self.pick_key_dim
        assert open_door.shape[0] == self.open_door_dim

        # Get the position of the potential next step
        # [left, right, top, bottom]
        next_pos = np.array(self.move_mapping[np.argmax(move)]) + self.agent_pos
        # get the contents of the cell in front of the agent
        next_cell = self.grid.get(*next_pos)

        reward = 0
        done = False

        # there is no priority between three actions, all of them can be executed at one single step
        # we should check P first, because we allow P and O at the same step
        if np.argmax(pick_key) == 1:
            curr_cell = self.grid.get(*self.agent_pos)
            # NOTE: we should check current position rather than next position
            if curr_cell and curr_cell.type == 'key':
                # update the grip map
                self.grid.set(*self.agent_pos, None)
                # dont update the key position of the state
                #self.state[self.agent_pos[1], self.agent_pos[0], 1] = 0.0
                self.have_key[0] = 0
                self.have_key[1] = 1

        if (np.argmax(self.have_key) == 1) and (np.argmax(open_door) == 1): # make sure we already have the key
            for i in self.move_mapping:
                sur_pos = self.agent_pos + np.array(i)
                sur_cell = self.grid.get(*sur_pos)
                # if sur_cell and sur_cell.type == 'door':
                if sur_cell and sur_cell.type == 'door' and (
                    sur_cell.is_locked and not sur_cell.is_open):
                    # we will make the door open and make it disappear in state
                    #self.grid.set(*sur_pos, Wall()) 
                    sur_cell.is_locked = False
                    sur_cell.is_open = True
                    # self.state[sur_pos[1], sur_pos[0], 2] = 0.0 # OLD STATE
                    self.state[sur_pos[0], 2] = 0.0 # NEW STATE
                    self.state[self.width + sur_pos[1], 2] = 0.0 # NEW STATE
                    # reward = 1 # OLD 

                    # if there is no other doors, finish
                    # if np.sum(self.state[:, :, 2]) == 0: # OLD STATE
                    if np.sum(self.state[:, 2]) == 0: # NEW STATE
                        done = True
                        reward = 1 # NEW
                        break

        if next_cell is None or next_cell.can_overlap():
            # self.state[self.agent_pos[1], self.agent_pos[0], 0] = 0.0 # OLD STATE
            self.state[self.agent_pos[0], 0] = 0.0 # NEW STATE
            self.state[self.width + self.agent_pos[1], 0] = 0.0 # NEW STATE
            self.agent_pos = next_pos
            # self.state[self.agent_pos[1], self.agent_pos[0], 0] = 1.0 # OLD STATE
            self.state[self.agent_pos[0], 0] = 1.0 # NEW STATE
            self.state[self.width + self.agent_pos[1], 0] = 1.0 # NEW STATE

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        return obs, reward, done, {}

    def gen_obs(self):
        '''
        print(self.agent_pos)
        print('agent_pos')
        print(self.state[:, :, 0])
        print('key_pos')
        print(self.state[:, :, 1])
        print('door_pos')
        print(self.state[:, :, 2])
        print('self.have_key', self.have_key)
        '''

        #obs_dict = {'have_key': self.have_key, 'state': self.state}
        # # NOTE: we should pack all states seperately to make the GNN part easier
        # agent_pos = self.state[:, :, 0].reshape(self.width*self.height) # OLD STATE 
        # key_pos = self.state[:, :, 1].reshape(self.width*self.height) # OLD STATE
        # door_pos = self.state[:, :, 2].reshape(self.width*self.height) # OLD STATE
        agent_pos = self.state[:, 0] # NEW STATE 
        key_pos = self.state[:, 1] # NEW STATE
        door_pos = self.state[:, 2] # NEW STATE

        obs = np.concatenate([agent_pos, key_pos, door_pos, self.have_key])
        # print(f"----Step count: {self.step_count}")
        # print(f"Agent position: {self.agent_pos}")
        # print(f"Key position: {np.where(key_pos==1)[0][0]%self.width, np.where(key_pos==1)[0][0]//self.height}")
        # if np.where(door_pos==1)[0].shape[0] != 0:
        #     print(f"Door position: {np.where(door_pos==1)[0][0]%self.width, np.where(door_pos==1)[0][0]//self.height}")
        # else:
        #     print(f"Door is open!")
        # print(f"Have key: {self.have_key}")
        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(tile_size, agent_pos=self.agent_pos, agent_dir=3, highlight_mask=vis_mask)
        return img

    def render(self, mode='human', close=False, tile_size=TILE_PIXELS):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            from .window import Window
            self.window = Window('Unlock')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(tile_size, self.agent_pos, self.agent_dir)
        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
