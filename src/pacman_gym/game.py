"""
Pac-Man core game engine.
Pure game logic with no rendering - reusable for both UI and Gym environment.
"""

import random
from enum import IntEnum
from typing import Optional

# Tile types
WALL = 0
DOT = 1       # small dot (10 pts)
POWER = 2     # big dot / power pellet (50 pts)
EMPTY = 3     # empty path
GHOST_HOUSE = 4
APPLE = 5     # magic apple

# Points
DOT_SCORE = 10
POWER_SCORE = 50
APPLE_SCORE = 100
GHOST_EAT_SCORE = 200

# Timing (in ticks)
APPLE_SPAWN_INTERVAL = 100   # ~14 seconds at FPS=7
INVINCIBLE_DURATION = 50     # ~7 seconds at FPS=7
GHOST_STUN_DURATION = 70     # ~10 seconds at FPS=7


class Direction(IntEnum):
    NONE = -1
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Direction vectors: (row_delta, col_delta)
DIR_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
    Direction.NONE: (0, 0),
}

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
    Direction.NONE: Direction.NONE,
}

# Classic-inspired Pac-Man maze (21 columns x 22 rows)
# W = wall, . = small dot, O = power pellet, _ = empty, G = ghost house, P = pacman start
MAZE_TEMPLATE = [
    "WWWWWWWWWWWWWWWWWWWWW",  # 0
    "W.........W.........W",  # 1
    "W.WWW.WWW.W.WWW.WWW.W",  # 2
    "WO.WW.WWW.W.WWW.WW.OW",  # 3
    "W...................W",  # 4
    "W.WW.W.WWWWWWW.W.WW.W",  # 5
    "W....W....W....W....W",  # 6
    "WWWW.WWWW.W.WWWW.WWWW",  # 7
    "WWWW.W_________W.WWWW",  # 8
    "WWWW.W_WGG_GGW_W.WWWW",  # 9  ghost house
    "_____._W_____W_._____",  # 10 tunnel
    "WWWW.W_WWWWWWW_W.WWWW",  # 11
    "WWWW.W_________W.WWWW",  # 12
    "WWWW.W.WWWWWWW.W.WWWW",  # 13
    "W...................W",  # 14
    "W.WWW.WWW.W.WWW.WWW.W",  # 15
    "WO..W.....P.....W..OW",  # 16 pac-man start
    "WW.WW.W.WWWWW.W.WW.WW",  # 17
    "W.....W...W...W.....W",  # 18
    "W.WWWWWWW.W.WWWWWWW.W",  # 19
    "W...................W",  # 20
    "WWWWWWWWWWWWWWWWWWWWW",  # 21
]


def _fuzz_maze(template: list[str], seed: int) -> list[str]:
    """Generate a maze variant by toggling some walls/dots.
    Preserves borders, ghost house (rows 7-13), tunnel, and start position."""
    rng = random.Random(seed)
    grid = [list(row) for row in template]
    rows, cols = len(grid), len(grid[0])
    protected_rows = {0, rows - 1, 7, 8, 9, 10, 11, 12, 13}

    for _ in range(rng.randint(10, 18)):
        r = rng.randint(2, rows - 3)
        c = rng.randint(2, cols - 3)
        if r in protected_rows:
            continue
        ch = grid[r][c]
        if ch == 'W':
            # Only remove wall if adjacent to >= 2 walkable tiles (creates shortcut)
            adj = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                      if 0 <= r + dr < rows and 0 <= c + dc < cols
                      and grid[r + dr][c + dc] not in ('W',))
            if adj >= 2:
                grid[r][c] = '.'
        elif ch == '.':
            # Only add wall if >= 3 walkable neighbors (won't block corridor)
            adj = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                      if 0 <= r + dr < rows and 0 <= c + dc < cols
                      and grid[r + dr][c + dc] not in ('W',))
            if adj >= 3:
                grid[r][c] = 'W'

    return [''.join(row) for row in grid]


MAZE_VARIANTS = [MAZE_TEMPLATE] + [_fuzz_maze(MAZE_TEMPLATE, seed=100 + i) for i in range(4)]


def parse_maze(template: list[str]) -> tuple[list[list[int]], list[tuple[int, int]], tuple[int, int]]:
    """Parse the maze template into a grid, ghost positions, and pacman start."""
    rows = len(template)
    cols = max(len(row) for row in template)

    grid = []
    ghost_positions = []
    pacman_start = (16, 10)  # default

    for r, row_str in enumerate(template):
        row = []
        for c in range(cols):
            if c < len(row_str):
                ch = row_str[c]
            else:
                ch = 'W'

            if ch == 'W':
                row.append(WALL)
            elif ch == '.':
                row.append(DOT)
            elif ch == 'O':
                row.append(POWER)
            elif ch == '_':
                row.append(EMPTY)
            elif ch == 'G':
                ghost_positions.append((r, c))
                row.append(GHOST_HOUSE)
            elif ch == 'P':
                pacman_start = (r, c)
                row.append(EMPTY)
            else:
                row.append(WALL)
        grid.append(row)

    return grid, ghost_positions, pacman_start


class Ghost:
    def __init__(self, row: int, col: int, color_id: int):
        self.row = row
        self.col = col
        self.start_row = row
        self.start_col = col
        self.color_id = color_id
        self.direction = Direction.NONE
        self.active = False
        self.release_timer = 0  # all ghosts start immediately
        self.stunned_timer = 0  # ticks remaining in stun

    @property
    def stunned(self) -> bool:
        return self.stunned_timer > 0

    def send_home(self):
        """Return ghost to ghost house and stun it."""
        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.NONE
        self.active = False
        self.stunned_timer = GHOST_STUN_DURATION
        self.release_timer = 0  # will re-activate after stun ends

    def reset(self):
        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.NONE
        self.active = False
        self.release_timer = 0
        self.stunned_timer = 0


class PacManGame:
    """Core Pac-Man game logic."""

    def __init__(self, maze_template: Optional[list[str]] = None, randomize_maze: bool = False):
        if maze_template is None:
            maze_template = MAZE_TEMPLATE
        self.maze_template = maze_template
        self.randomize_maze = randomize_maze
        self.reset()

    def reset(self):
        if self.randomize_maze:
            self.maze_template = random.choice(MAZE_VARIANTS)
        """Reset the game to initial state. Returns observation-like info."""
        self.grid, ghost_starts, self.pacman_start = parse_maze(self.maze_template)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0

        # Pac-Man state
        self.pac_row, self.pac_col = self.pacman_start
        self.pac_direction = Direction.LEFT
        self.desired_direction = Direction.LEFT

        # Ghost state
        self.ghosts: list[Ghost] = []
        for i, (gr, gc) in enumerate(ghost_starts):
            if i < 4:
                self.ghosts.append(Ghost(gr, gc, i))

        # Game state
        self.score = 0
        self.game_over = False
        self.tick_count = 0
        self.total_dots = sum(1 for r in self.grid for t in r if t in (DOT, POWER))
        self.dots_eaten = 0
        self.won = False

        # Apple & invincibility
        self.apple_timer = APPLE_SPAWN_INTERVAL
        self.apple_pos: Optional[tuple[int, int]] = None
        self.invincible_timer = 0

        # Cache empty positions for apple spawning
        self._empty_positions: list[tuple[int, int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (DOT, EMPTY):
                    self._empty_positions.append((r, c))

        return self._get_state()

    def _is_walkable(self, row: int, col: int) -> bool:
        """Check if a tile can be walked on."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            # Allow tunnel wrapping
            return True
        return self.grid[row][col] != WALL

    def _wrap(self, row: int, col: int) -> tuple[int, int]:
        """Wrap coordinates for tunnel."""
        return row % self.rows, col % self.cols

    def _can_move(self, row: int, col: int, direction: Direction) -> bool:
        """Check if movement in a direction is possible from given position."""
        if direction == Direction.NONE:
            return False
        dr, dc = DIR_DELTA[direction]
        nr, nc = self._wrap(row + dr, col + dc)
        return self._is_walkable(nr, nc)

    def _get_available_directions(self, row: int, col: int, exclude: Optional[Direction] = None) -> list[Direction]:
        """Get all walkable directions from a position."""
        dirs = []
        for d in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            if d == exclude:
                continue
            if self._can_move(row, col, d):
                dirs.append(d)
        return dirs

    def set_direction(self, direction: Direction):
        """Set the desired direction for Pac-Man (called by player input or agent)."""
        self.desired_direction = direction

    @property
    def invincible(self) -> bool:
        return self.invincible_timer > 0

    def step(self) -> tuple[dict, int, bool]:
        """
        Advance the game by one tick.
        Returns: (state, reward, done)
        """
        if self.game_over:
            return self._get_state(), 0, True

        reward = 0
        self.tick_count += 1

        # Tick down invincibility
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        # Spawn apple periodically
        self._tick_apple()

        # Move Pac-Man
        reward += self._move_pacman()

        # Move ghosts (with anti-overlap logic)
        self._move_ghosts()

        # Check pac-man vs ghost collision
        reward += self._check_collision()
        if self.game_over:
            return self._get_state(), reward, True

        # Check win
        if self.dots_eaten >= self.total_dots:
            self.game_over = True
            self.won = True
            return self._get_state(), reward + 500, True

        return self._get_state(), reward, False

    def _tick_apple(self):
        """Handle magic apple spawning."""
        if self.apple_pos is not None:
            return  # apple already on map
        self.apple_timer -= 1
        if self.apple_timer <= 0:
            # Spawn apple on a random empty tile
            candidates = [
                (r, c) for r, c in self._empty_positions
                if self.grid[r][c] == EMPTY
                and (r, c) != (self.pac_row, self.pac_col)
            ]
            if candidates:
                ar, ac = random.choice(candidates)
                self.grid[ar][ac] = APPLE
                self.apple_pos = (ar, ac)
            self.apple_timer = APPLE_SPAWN_INTERVAL

    def _move_pacman(self) -> int:
        """Move Pac-Man and eat dots. Returns reward."""
        # Try desired direction first
        if self.desired_direction != Direction.NONE and self._can_move(self.pac_row, self.pac_col, self.desired_direction):
            self.pac_direction = self.desired_direction

        # If current direction is blocked, auto-turn
        if self.pac_direction != Direction.NONE and not self._can_move(self.pac_row, self.pac_col, self.pac_direction):
            reverse = OPPOSITE[self.pac_direction]
            # First try non-reverse directions (corner auto-turn)
            turned = False
            for d in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                if d != reverse and self._can_move(self.pac_row, self.pac_col, d):
                    self.pac_direction = d
                    turned = True
                    break
            # Dead end: reverse is the only option
            if not turned and self._can_move(self.pac_row, self.pac_col, reverse):
                self.pac_direction = reverse

        # Move in current direction
        if self.pac_direction != Direction.NONE and self._can_move(self.pac_row, self.pac_col, self.pac_direction):
            dr, dc = DIR_DELTA[self.pac_direction]
            self.pac_row, self.pac_col = self._wrap(self.pac_row + dr, self.pac_col + dc)

        # Eat items on current tile
        reward = 0
        tile = self.grid[self.pac_row][self.pac_col]
        if tile == DOT:
            self.grid[self.pac_row][self.pac_col] = EMPTY
            self.score += DOT_SCORE
            self.dots_eaten += 1
            reward = DOT_SCORE
        elif tile == POWER:
            self.grid[self.pac_row][self.pac_col] = EMPTY
            self.score += POWER_SCORE
            self.dots_eaten += 1
            reward = POWER_SCORE
        elif tile == APPLE:
            self.grid[self.pac_row][self.pac_col] = EMPTY
            self.score += APPLE_SCORE
            self.apple_pos = None
            self.invincible_timer = INVINCIBLE_DURATION
            reward = APPLE_SCORE

        return reward

    def _ghost_occupied(self, row: int, col: int, exclude: Ghost) -> bool:
        """Check if another active ghost is at the given position."""
        for g in self.ghosts:
            if g is exclude:
                continue
            if g.active and not g.stunned and g.row == row and g.col == col:
                return True
        return False

    def _move_ghosts(self):
        """Move all ghosts. Ghosts cannot reverse; at intersections they pick randomly.
        Ghosts avoid moving into a tile occupied by another ghost."""
        for ghost in self.ghosts:
            # Handle stunned ghosts in ghost house
            if ghost.stunned:
                ghost.stunned_timer -= 1
                continue

            # Handle release timer
            if not ghost.active:
                ghost.release_timer -= 1
                if ghost.release_timer <= 0:
                    ghost.active = True
                    ghost.row = 8
                    ghost.col = 10
                    ghost.direction = random.choice([Direction.LEFT, Direction.RIGHT])
                continue

            # Ghost moves every 2 ticks out of 3 (slower than pac-man)
            if self.tick_count % 3 == 0:
                continue

            reverse_dir = OPPOSITE[ghost.direction]

            # Get available directions (excluding reverse)
            available = self._get_available_directions(ghost.row, ghost.col, exclude=reverse_dir)

            if not available:
                available = [reverse_dir] if self._can_move(ghost.row, ghost.col, reverse_dir) else []

            if not available:
                continue

            # Filter out directions that lead into another ghost
            safe = []
            for d in available:
                dr, dc = DIR_DELTA[d]
                nr, nc = self._wrap(ghost.row + dr, ghost.col + dc)
                if not self._ghost_occupied(nr, nc, exclude=ghost):
                    safe.append(d)

            candidates = safe if safe else available

            # Weight directions: prefer moving away from other ghosts (spread out)
            weights = []
            for d in candidates:
                dr, dc = DIR_DELTA[d]
                nr, nc = self._wrap(ghost.row + dr, ghost.col + dc)
                # Sum of inverse distances to other active ghosts
                crowd_penalty = 0.0
                for other in self.ghosts:
                    if other is ghost or not other.active or other.stunned:
                        continue
                    dist = abs(nr - other.row) + abs(nc - other.col)
                    crowd_penalty += 1.0 / max(dist, 1)
                # Lower penalty = higher weight (farther from others is better)
                weights.append(1.0 / (1.0 + crowd_penalty * 3.0))

            # Weighted random choice
            total = sum(weights)
            r = random.random() * total
            cumul = 0.0
            chosen = candidates[0]
            for d, w in zip(candidates, weights):
                cumul += w
                if r <= cumul:
                    chosen = d
                    break

            ghost.direction = chosen
            dr, dc = DIR_DELTA[chosen]
            ghost.row, ghost.col = self._wrap(ghost.row + dr, ghost.col + dc)

    def _check_collision(self) -> int:
        """Check Pac-Man vs ghost. Returns reward (negative if death, positive if eating ghost)."""
        reward = 0
        for ghost in self.ghosts:
            if ghost.active and not ghost.stunned and ghost.row == self.pac_row and ghost.col == self.pac_col:
                if self.invincible:
                    # Eat the ghost!
                    ghost.send_home()
                    self.score += GHOST_EAT_SCORE
                    reward += GHOST_EAT_SCORE
                else:
                    self.game_over = True
                    reward = -1000
                    return reward
        return reward

    def _get_state(self) -> dict:
        """Get the current game state as a dictionary."""
        return {
            "grid": [row[:] for row in self.grid],
            "pac_pos": (self.pac_row, self.pac_col),
            "pac_dir": self.pac_direction,
            "ghosts": [(g.row, g.col, g.active, g.stunned, int(g.direction)) for g in self.ghosts],
            "score": self.score,
            "game_over": self.game_over,
            "won": self.won,
            "dots_remaining": self.total_dots - self.dots_eaten,
            "tick": self.tick_count,
            "invincible": self.invincible,
            "invincible_timer": self.invincible_timer,
            "apple_pos": self.apple_pos,
        }
