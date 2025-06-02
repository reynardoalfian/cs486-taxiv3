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
        self.passenger_in_taxi: int | None = None  # 0,1, or None (stores index 0 or 1)
        self.obstacles = {(1, 1), (3, 3)} #locations of obstacles
        self.passengers_delivered = [False, False] # [passenger1_delivered, passenger2_delivered]

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
        self.passengers_delivered = [False, False]
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
        reward, terminated = -1, False # Default reward is -1 per step
        r, c, p1, d1, p2, d2 = self.decode6(self.s)

        # Store current taxi, passenger locations, and passenger in taxi status for reward shaping
        old_r, old_c = r, c
        old_p1_loc, old_p2_loc = p1, p2 # Store passenger locations BEFORE action
        old_passenger_in_taxi = self.passenger_in_taxi

        # Movement and illegal move penalties
        r, c, illegal = self._move(r, c, action)
        reward += illegal

        if action == 4: # Pickup action
            if self.passenger_in_taxi is None: # Only allow pickup if taxi is empty
                if p1 < 4 and (r, c) == self.locs[p1] and not self.passengers_delivered[0]:
                    p1, self.passenger_in_taxi = 4, 0
                    reward += 10 # Increased reward for successful pickup
                elif p2 < 4 and (r, c) == self.locs[p2] and not self.passengers_delivered[1]:
                    p2, self.passenger_in_taxi = 4, 1
                    reward += 10 # Increased reward for successful pickup
                else:
                    reward = -10 # Illegal pickup (no passenger at location or already delivered)
            else: # Taxi already has a passenger
                reward = -10 # Illegal pickup
        elif action == 5: # Dropoff action
            if self.passenger_in_taxi == 0 and p1 == 4 and (r, c) == self.locs[d1]:
                p1, self.passenger_in_taxi = d1, None
                self.passengers_delivered[0] = True
                reward = +20 # Large reward for delivering passenger 1
            elif self.passenger_in_taxi == 1 and p2 == 4 and (r, c) == self.locs[d2]:
                p2, self.passenger_in_taxi = d2, None
                self.passengers_delivered[1] = True
                reward = +20 # Large reward for delivering passenger 2
            else:
                reward = -10 # Illegal dropoff

        self.s = self.encode(r, c, p1, d1, p2, d2)
        self.state = self.s

        # --- Reward Shaping for movement progress ---
        # Reward for moving closer to a passenger
        if old_passenger_in_taxi is None: # Only if taxi is empty
            # Check for passenger 1
            if old_p1_loc < 4 and not self.passengers_delivered[0]:
                target_row, target_col = self.locs[old_p1_loc]
                old_dist = abs(target_row - old_r) + abs(target_col - old_c)
                new_dist = abs(target_row - r) + abs(target_col - c)
                if new_dist < old_dist:
                    reward += 1 # More substantial reward for moving closer to an available passenger
            # Check for passenger 2
            if old_p2_loc < 4 and not self.passengers_delivered[1]:
                target_row, target_col = self.locs[old_p2_loc]
                old_dist = abs(target_row - old_r) + abs(target_col - old_c)
                new_dist = abs(target_row - r) + abs(target_col - c)
                if new_dist < old_dist:
                    reward += 1 # More substantial reward for moving closer to an available passenger
        else: # If a passenger is in the taxi, reward for moving closer to *their* destination
            if old_passenger_in_taxi == 0 and not self.passengers_delivered[0]: # Passenger 1 in taxi
                target_row, target_col = self.locs[d1]
                old_dist = abs(target_row - old_r) + abs(target_col - old_c)
                new_dist = abs(target_row - r) + abs(target_col - c)
                if new_dist < old_dist:
                    reward += 1 # More substantial reward for moving closer to the destination
            elif old_passenger_in_taxi == 1 and not self.passengers_delivered[1]: # Passenger 2 in taxi
                target_row, target_col = self.locs[d2]
                old_dist = abs(target_row - old_r) + abs(target_col - old_c)
                new_dist = abs(target_row - r) + abs(target_col - c)
                if new_dist < old_dist:
                    reward += 1 # More substantial reward for moving closer to the destination

        # Episode terminates only if *both* passengers are delivered
        if self.passengers_delivered[0] and self.passengers_delivered[1]:
            terminated = True
            # A final bonus for completing the overall task
            reward += 100 # Significant bonus for successful completion

        return int(self.s), reward, terminated, False, {}

    def _move(self, row: int, col: int, action: int):
        new_row, new_col = row, col
        illegal = 0

        if action == 0 and row < 4:  # Move down
            new_row = row + 1
        elif action == 1 and row > 0:  # Move up
            new_row = row - 1
        elif action == 2 and col < 4 and self.desc[1 + row, 2 * col + 2] == b":":  # Move right
            new_col = col + 1
        elif action == 3 and col > 0 and self.desc[1 + row, 2 * col] == b":":  # Move left
            new_col = col - 1
        else:
            # Invalid movement (hitting wall or boundary)
            if action in (0, 1, 2, 3):  # Only penalize actual movement actions
                illegal = -10 # Penalty for hitting a wall

        # Check if new position hits an obstacle
        if (new_row, new_col) in self.obstacles:
            illegal = -10 # Penalty for hitting an obstacle
            return row, col, illegal  # Stay in original position

        return new_row, new_col, illegal

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
        if p1<4: # Passenger 1 is at a location
            cx,cy = pos(p1)
            pygame.draw.circle(self.window, BLACK, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10)
        elif self.passengers_delivered[0]: # Passenger 1 delivered, render at destination
            cx,cy = pos(d1)
            pygame.draw.circle(self.window, BLACK, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10, width=2) # outline
        if p2<4: # Passenger 2 is at a location
            cx,cy = pos(p2)
            pygame.draw.circle(self.window, ORANGE, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10)
        elif self.passengers_delivered[1]: # Passenger 2 delivered, render at destination
            cx,cy = pos(d2)
            pygame.draw.circle(self.window, ORANGE, (cx*cell_w+cell_w//2, cy*cell_h+cell_h//2),10, width=2) # outline

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