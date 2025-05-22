import pygame
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.taxi import TaxiEnv

# -----------------------------------------------------------------------------
#  Environment registration (so `gym.make()` can find it)
# -----------------------------------------------------------------------------
gym.register(
    id="TaxiTwoPassenger-v0",
    entry_point="multi_taxi:TaxiTwoPassengerEnv",
    max_episode_steps=200,
    reward_threshold=40,  # 2 passengers × +20 each
)

# -----------------------------------------------------------------------------
#  TaxiTwoPassengerEnv
# -----------------------------------------------------------------------------
class TaxiTwoPassengerEnv(TaxiEnv):
    """5 × 5 grid, two passengers, otherwise same rules as Taxi-v3."""

    # 25 cells × (5 locs × 4 dests)²  = 10 000 states
    observation_space: spaces.Discrete = spaces.Discrete(25 * 5 * 4 * 5 * 4)

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)
        self.observation_space = TaxiTwoPassengerEnv.observation_space
        self.window, self.clock = None, None
        self.passenger_in_taxi: int | None = None  # 0,1, or None
        self.obstacles = {(1, 1), (3, 3)} //locations of obstacles

    @staticmethod
    def encode(r: int, c: int, p1: int, d1: int, p2: int | None = None, d2: int | None = None) -> int:
        """If p2/d2 omitted => fall back to Taxi-v3 encoding (500 states)."""
        if p2 is None or d2 is None:
            i = r; i = i * 5 + c; i = i * 5 + p1; i = i * 4 + d1
            return i
        i = r; i = i * 5 + c; i = i * 5 + p1; i = i * 4 + d1; i = i * 5 + p2; i = i * 4 + d2
        return i

    @staticmethod
    def decode(i: int):
        dest = i % 4; i //= 4
        passenger = i % 5; i //= 5
        c = i % 5; i //= 5
        r = i
        return r, c, passenger, dest

    @staticmethod
    def decode6(i: int):
        d2 = i % 4; i //= 4
        p2 = i % 5; i //= 5
        d1 = i % 4; i //= 4
        p1 = i % 5; i //= 5
        c = i % 5; i //= 5
        r = i
        return r, c, p1, d1, p2, d2

    def _generate_random_state(self, rng):
        r, c = rng.integers(5, size=2)
        p1, p2 = rng.integers(4, size=2)
        d1, d2 = rng.integers(4, size=2)
        self.passenger_in_taxi = None
        return self.encode(r, c, p1, d1, p2, d2)

    def reset(self, *, seed: int | None = None, options=None):
        self.passenger_in_taxi = None
        orig_mode, self.render_mode = self.render_mode, None
        super().reset(seed=seed)
        self.render_mode = orig_mode
        self.s = self._generate_random_state(self.np_random)
        self.state = self.s
        if self.render_mode == "human":
            self._render_gui("human")
        return int(self.s), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward, terminated = -1, False
        r, c, p1, d1, p2, d2 = self.decode6(self.s)
        r, c, illegal = self._move(r, c, action)
        reward += illegal
        if action == 4:
            if p1 < 4 and (r, c) == self.locs[p1]:
                p1, self.passenger_in_taxi = 4, 0
            elif p2 < 4 and (r, c) == self.locs[p2]:
                p2, self.passenger_in_taxi = 4, 1
            else:
                reward = -10
        elif action == 5:
            if self.passenger_in_taxi == 0 and p1 == 4 and (r, c) == self.locs[d1]:
                p1, reward, self.passenger_in_taxi = d1, +20, None
            elif self.passenger_in_taxi == 1 and p2 == 4 and (r, c) == self.locs[d2]:
                p2, reward, self.passenger_in_taxi = d2, +20, None
            else:
                reward = -10
        self.s = self.encode(r, c, p1, d1, p2, d2)
        self.state = self.s
        if p1 == d1 and p2 == d2:
            terminated = True
        return int(self.s), reward, terminated, False, {}

    def _move(self, row: int, col: int, action: int):
        new_row, new_col = row, col
        illegal = 0
        if action == 0 and row < 4:
            row += 1
        elif action == 1 and row > 0:
            row -= 1
        elif action == 2 and col < 4 and self.desc[1 + row, 2 * col + 2] == b":":
            col += 1
        elif action == 3 and col > 0 and self.desc[1 + row, 2 * col] == b":":
            col -= 1
        else:
            if action in (2, 3):
                illegal = -10
        if (new_row, new_col) in self.obstacles:
            illegal = -10
            return row, col, illegal
        
        return row, col, illegal

    def _render_gui(self, mode):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((700, 400))
            pygame.display.set_caption("Taxi – Two Passengers")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        
        WHITE, BLACK = (255,255,255), (0,0,0)
        YELLOW, ORANGE = (255,255,0), (255,165,0)
        RED, GREEN, BLUE = (255,0,0), (0,200,0), (0,0,255)
        cell_w, cell_h, border = 100, 80, 4
        self.window.fill(WHITE)
        for c in range(6): pygame.draw.line(self.window, BLACK, (c*cell_w,0), (c*cell_w,cell_h*5),border)
        for r in range(6): pygame.draw.line(self.window, BLACK, (0,r*cell_h), (cell_w*5,r*cell_h),border)

        GRAY = (160, 160, 160)
        for (r, c) in self.obstacles:
            obs_rect = pygame.Rect(c*cell_w+10, r*cell_h+10, cell_w-20, cell_h-20)
            pygame.draw.rect(self.window, GRAY, obs_rect)
        row, col, p1, d1, p2, d2 = self.decode6(self.s)
        dest_colors = [RED, GREEN, YELLOW, BLUE]
        for idx,(rr,cc) in enumerate(self.locs):
            rect = pygame.Rect(cc*cell_w+10, rr*cell_h+10, cell_w-20, cell_h-20)
            pygame.draw.rect(self.window, dest_colors[idx], rect, width=border)
        def pos(i): return self.locs[i][1], self.locs[i][0]
        if p1<4:
            cx,cy = pos(p1)
            pygame.draw.circle(self.window, BLACK, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10)
        if p2<4:
            cx,cy = pos(p2)
            pygame.draw.circle(self.window, ORANGE, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10)
        taxi_rect = pygame.Rect(col*cell_w+20, row*cell_h+20, cell_w-40, cell_h-40)
        pygame.draw.rect(self.window, YELLOW, taxi_rect)
        if self.passenger_in_taxi==0:
            pygame.draw.circle(self.window, BLACK, taxi_rect.center,10)
        elif self.passenger_in_taxi==1:
            pygame.draw.circle(self.window, ORANGE, taxi_rect.center,10)
        # legend
        font = pygame.font.SysFont(None,18)
        legends = [
            ("Taxi","Yellow rectangle"),
            ("Passenger 1","Black circle"),
            ("Passenger 2","Orange circle"),
            ("Destinations","Colored squares"),
            ("Obstacle", "Gray square"),

        ]
        for i,(title,desc) in enumerate(legends):
            txt = font.render(f"{title}: {desc}", True, BLACK)
            self.window.blit(txt, (520,20 + i*25))
        pygame.event.pump(); self.clock.tick(15); pygame.display.flip()
        if mode=="rgb_array": return np.transpose(pygame.surfarray.array3d(self.window),(1,0,2))