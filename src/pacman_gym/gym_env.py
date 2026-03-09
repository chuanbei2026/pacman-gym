"""
Pac-Man Gymnasium environment for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import (
    PacManGame, Direction, WALL, DOT, POWER, EMPTY, GHOST_HOUSE, APPLE,
)

# Channel encoding for observation grid
TILE_CHANNEL_MAP = {
    WALL: 0,
    DOT: 1,
    POWER: 2,
    EMPTY: 3,
    GHOST_HOUSE: 3,
    APPLE: 4,
}


class PacManEnv(gym.Env):
    """
    Pac-Man as a Gymnasium environment.

    Observation: a flat feature vector encoding:
      - Grid state (each cell: wall/dot/power/empty)
      - Pac-Man position (row, col normalized)
      - Ghost positions + active flags
      - Current direction

    Action space: Discrete(4) - UP, RIGHT, DOWN, LEFT

    Reward:
      - +10 for eating a small dot
      - +50 for eating a power pellet
      - -100 for dying (ghost collision)
      - +500 for eating all dots (win)
      - -1 per step (encourage efficiency)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, max_steps=2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Create game with maze randomization for training diversity
        self._game = PacManGame(randomize_maze=True)
        self._game.reset()
        self.rows = self._game.rows
        self.cols = self._game.cols

        # Action space: UP=0, RIGHT=1, DOWN=2, LEFT=3
        self.action_space = spaces.Discrete(4)

        # Observation:
        #   local_grid 15x15 with walls+dots+ghosts encoded: 225
        #   ghost_info 4 ghosts * 6 (rel_r, rel_c, dist, dir, active, stunned): 24
        #   pac_dir: 1, invincible: 1
        #   dot_dirs (4 directions): 4, nearest_dot_rel: 2, dots_remaining: 1
        # Total: 225 + 24 + 1 + 1 + 4 + 2 + 1 = 258
        obs_size = 258
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self._step_count = 0

        # For pygame rendering
        self._screen = None
        self._clock = None

    def _get_obs(self) -> np.ndarray:
        """Convert game state to a compact observation vector (258 features).

        Layout:
          - local_grid 15x15 = 225 (terrain + ghost overlay around Pac-Man)
          - ghost_info 4 x 6 = 24  (rel_row, rel_col, dist, direction, active, stunned)
          - pac_dir: 1
          - invincible: 1
          - dot_dirs: 4  (distance to nearest dot in each direction via corridor scan)
          - nearest_dot_rel: 2  (relative row/col to nearest dot)
          - dots_remaining_ratio: 1
        """
        state = self._game._get_state()
        grid = state["grid"]
        pr, pc = state["pac_pos"]
        max_dist = self.rows + self.cols

        # --- Build ghost position lookup for 15x15 overlay ---
        ghost_cells = {}  # (r, c) -> value
        for gr, gc, active, stunned, gdir in state["ghosts"]:
            if active:
                ghost_cells[(gr, gc)] = -0.5 if stunned else -1.0

        # --- Local 15x15 grid with terrain + ghost overlay ---
        TILE_VAL = {WALL: 0.0, DOT: 0.4, POWER: 0.6, EMPTY: 0.2, GHOST_HOUSE: 0.2, APPLE: 0.8}
        local_grid = []
        for dr in range(-7, 8):
            for dc in range(-7, 8):
                r = (pr + dr) % self.rows
                c = (pc + dc) % self.cols
                if (r, c) in ghost_cells:
                    local_grid.append(ghost_cells[(r, c)])
                else:
                    local_grid.append(TILE_VAL.get(grid[r][c], 0.0))

        # --- Ghost info: relative position, distance, direction, active, stunned ---
        ghost_info = []
        for gr, gc, active, stunned, gdir in state["ghosts"]:
            rel_r = (gr - pr) / self.rows
            rel_c = (gc - pc) / self.cols
            dist = (abs(gr - pr) + abs(gc - pc)) / max_dist
            g_direction = gdir / 3.0 if gdir >= 0 else -1.0
            ghost_info.extend([
                rel_r, rel_c, dist, g_direction,
                1.0 if active else 0.0,
                1.0 if stunned else 0.0,
            ])
        while len(ghost_info) < 24:
            ghost_info.extend([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])

        # --- Pac-Man direction ---
        pac_dir = [state["pac_dir"] / 3.0 if state["pac_dir"] >= 0 else -1.0]

        # --- Invincible ---
        inv = [1.0 if state.get("invincible", False) else 0.0]

        # --- Dot direction scan: walk each direction until wall, find nearest dot ---
        dir_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dot_dirs = []
        for ddr, ddc in dir_deltas:
            found = 0.0
            for step in range(1, max(self.rows, self.cols)):
                r = (pr + ddr * step) % self.rows
                c = (pc + ddc * step) % self.cols
                tile = grid[r][c]
                if tile == WALL:
                    break
                if tile in (DOT, POWER):
                    found = 1.0 - step / max_dist
                    break
            dot_dirs.append(found)

        # --- Nearest dot relative position ---
        best_dot_dist = max_dist
        best_dr, best_dc = 0.0, 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] in (DOT, POWER):
                    d = abs(r - pr) + abs(c - pc)
                    if d < best_dot_dist:
                        best_dot_dist = d
                        best_dr = (r - pr) / self.rows
                        best_dc = (c - pc) / self.cols
        nearest_dot_rel = [best_dr, best_dc]

        # --- Dots remaining ratio ---
        dots_ratio = [state["dots_remaining"] / max(self._game.total_dots, 1)]

        obs = np.array(
            local_grid + ghost_info + pac_dir + inv
            + dot_dirs + nearest_dot_rel + dots_ratio,
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._game.reset()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        direction = Direction(action)
        self._game.set_direction(direction)
        state, reward, done = self._game.step()
        self._step_count += 1

        # Time-based penalty that grows over time to discourage stalling
        reward -= 1 + self._step_count / 100.0

        # Truncate if max steps exceeded
        truncated = self._step_count >= self.max_steps
        if truncated:
            done = True

        obs = self._get_obs()
        info = {
            "score": state["score"],
            "dots_remaining": state["dots_remaining"],
            "won": state["won"],
            "steps": self._step_count,
        }

        return obs, float(reward), done, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_human(self):
        try:
            import pygame
            from .main import (
                draw_maze, draw_pacman, draw_ghost, draw_score,
                TILE_SIZE, SCORE_BAR_HEIGHT, GHOST_COLORS, BG_COLOR,
            )
        except ImportError:
            return

        if self._screen is None:
            pygame.init()
            width = self.cols * TILE_SIZE
            height = self.rows * TILE_SIZE + SCORE_BAR_HEIGHT
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Pac-Man (RL Agent)")
            self._clock = pygame.time.Clock()

        state = self._game._get_state()
        width = self.cols * TILE_SIZE

        self._screen.fill(BG_COLOR)
        draw_score(self._screen, state["score"], width,
                   state.get("invincible", False), state.get("invincible_timer", 0))
        draw_maze(self._screen, state["grid"], SCORE_BAR_HEIGHT)

        for i, (gr, gc, active, stunned, *_) in enumerate(state["ghosts"]):
            color = GHOST_COLORS[i] if i < len(GHOST_COLORS) else (255, 255, 255)
            draw_ghost(self._screen, gr, gc, color, active, stunned, state["tick"], SCORE_BAR_HEIGHT)

        pr, pc = state["pac_pos"]
        draw_pacman(self._screen, pr, pc, state["pac_dir"], state["tick"], SCORE_BAR_HEIGHT,
                    invincible=state.get("invincible", False))

        pygame.display.flip()
        self._clock.tick(30)

        # Process events to prevent freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _render_rgb(self):
        # Simple grid-based RGB rendering without pygame
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        state = self._game._get_state()

        for r in range(self.rows):
            for c in range(self.cols):
                tile = state["grid"][r][c]
                if tile == WALL:
                    img[r, c] = [33, 33, 222]
                elif tile == DOT:
                    img[r, c] = [255, 183, 174]
                elif tile == POWER:
                    img[r, c] = [255, 255, 174]

        # Draw pac-man
        pr, pc = state["pac_pos"]
        img[pr, pc] = [255, 255, 0]

        # Draw ghosts
        ghost_colors = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]
        for i, (gr, gc, active) in enumerate(state["ghosts"]):
            if active and 0 <= gr < self.rows and 0 <= gc < self.cols:
                img[gr, gc] = ghost_colors[i] if i < len(ghost_colors) else (255, 255, 255)

        return img

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
