"""
Pac-Man game with Pygame UI.
Run: python3 -m pacman_gym
"""

import sys
import math
import pygame
from .game import (
    PacManGame, Direction, WALL, DOT, POWER, EMPTY, GHOST_HOUSE, APPLE,
)

# Display settings
TILE_SIZE = 28
SCORE_BAR_HEIGHT = 50
FPS = 5

# Colors
BLACK = (0, 0, 0)
BLUE = (33, 33, 222)
DARK_BLUE = (20, 20, 120)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
DOT_COLOR = (255, 183, 174)
POWER_COLOR = (255, 183, 174)

GHOST_COLORS = [
    (255, 0, 0),       # Blinky - red
    (255, 184, 255),    # Pinky - pink
    (0, 255, 255),      # Inky - cyan
    (255, 184, 82),     # Clyde - orange
]

APPLE_COLOR = (0, 220, 0)
APPLE_STEM = (139, 90, 43)
STUNNED_COLOR = (100, 100, 200)

WALL_COLOR = BLUE
BG_COLOR = BLACK


def _is_wall(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True
    return grid[r][c] == WALL


def draw_maze(surface: pygame.Surface, grid: list[list[int]], offset_y: int):
    """Draw the maze walls and dots with connected wall edges."""
    rows = len(grid)
    cols = len(grid[0])

    for r in range(rows):
        for c in range(cols):
            x = c * TILE_SIZE
            y = r * TILE_SIZE + offset_y
            tile = grid[r][c]

            if tile == WALL:
                # Solid filled wall block with slight inset for grid look
                pygame.draw.rect(surface, WALL_COLOR, (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2))

            elif tile == DOT:
                cx = x + TILE_SIZE // 2
                cy = y + TILE_SIZE // 2
                pygame.draw.circle(surface, DOT_COLOR, (cx, cy), 2)
            elif tile == POWER:
                cx = x + TILE_SIZE // 2
                cy = y + TILE_SIZE // 2
                pygame.draw.circle(surface, POWER_COLOR, (cx, cy), 6)
            elif tile == APPLE:
                cx = x + TILE_SIZE // 2
                cy = y + TILE_SIZE // 2
                # Apple body
                pygame.draw.circle(surface, APPLE_COLOR, (cx, cy + 2), 8)
                # Stem
                pygame.draw.line(surface, APPLE_STEM, (cx, cy - 5), (cx + 2, cy - 8), 2)
                # Leaf
                pygame.draw.ellipse(surface, (0, 180, 0), (cx + 1, cy - 9, 6, 4))


def draw_pacman(surface: pygame.Surface, row: int, col: int, direction: Direction,
                tick: int, offset_y: int, invincible: bool = False):
    """Draw Pac-Man with mouth animation."""
    cx = col * TILE_SIZE + TILE_SIZE // 2
    cy = row * TILE_SIZE + TILE_SIZE // 2 + offset_y
    radius = TILE_SIZE // 2 - 1

    # Invincible glow
    if invincible:
        glow_r = radius + 4 + (tick % 3)
        glow_color = (255, 255, 100, 80) if tick % 2 == 0 else (255, 200, 50, 80)
        glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_r, glow_r), glow_r)
        surface.blit(glow_surf, (cx - glow_r, cy - glow_r))

    # Mouth animation
    mouth_angle = abs(math.sin(tick * 0.5)) * 45

    # Direction to start angle
    angle_map = {
        Direction.RIGHT: 0,
        Direction.UP: 90,
        Direction.LEFT: 180,
        Direction.DOWN: 270,
        Direction.NONE: 0,
    }
    base_angle = angle_map.get(direction, 0)

    start = math.radians(base_angle + mouth_angle)
    end = math.radians(base_angle + 360 - mouth_angle)

    # Draw pac-man as a pie shape
    points = [(cx, cy)]
    steps = 30
    for i in range(steps + 1):
        angle = start + (end - start) * i / steps
        px = cx + radius * math.cos(angle)
        py = cy - radius * math.sin(angle)
        points.append((px, py))
    points.append((cx, cy))

    if len(points) > 2:
        pygame.draw.polygon(surface, YELLOW, points)


def draw_ghost(surface: pygame.Surface, row: int, col: int, color: tuple,
               active: bool, stunned: bool, tick: int, offset_y: int):
    """Draw a ghost."""
    cx = col * TILE_SIZE + TILE_SIZE // 2
    cy = row * TILE_SIZE + TILE_SIZE // 2 + offset_y
    radius = TILE_SIZE // 2 - 2

    if stunned:
        # Stunned: blinking blue/white
        color = STUNNED_COLOR if tick % 4 < 2 else (220, 220, 255)
    elif not active:
        color = tuple(c // 2 for c in color)

    # Ghost body - semicircle top + rectangle bottom
    pygame.draw.circle(surface, color, (cx, cy - 2), radius)
    rect = pygame.Rect(cx - radius, cy - 2, radius * 2, radius + 2)
    pygame.draw.rect(surface, color, rect)

    # Wavy skirt bottom
    skirt_y = cy + radius - 1
    wave = tick % 2
    for i in range(4):
        sx = cx - radius + i * (radius * 2 // 4) + (wave * 2)
        if sx < cx - radius or sx > cx + radius:
            continue
        pygame.draw.circle(surface, BG_COLOR, (sx, skirt_y), 2)

    # Eyes
    eye_offset_x = radius // 3 + 1
    eye_y = cy - 3
    for ex in [cx - eye_offset_x, cx + eye_offset_x]:
        pygame.draw.circle(surface, WHITE, (ex, eye_y), 4)
        pygame.draw.circle(surface, (33, 33, 200), (ex + 1, eye_y + 1), 2)


def draw_score(surface: pygame.Surface, score: int, width: int,
               invincible: bool = False, inv_timer: int = 0):
    """Draw the score bar at the top."""
    font = pygame.font.SysFont("arial", 24, bold=True)
    text = font.render(f"SCORE: {score}", True, WHITE)
    surface.blit(text, (10, 12))

    if invincible:
        inv_text = font.render(f"INVINCIBLE! {inv_timer}", True, (255, 255, 100))
        surface.blit(inv_text, (width - inv_text.get_width() - 10, 12))


def draw_game_over(surface: pygame.Surface, won: bool, score: int, width: int, height: int):
    """Draw game over overlay."""
    # Semi-transparent overlay
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))

    font_big = pygame.font.SysFont("arial", 48, bold=True)
    font_small = pygame.font.SysFont("arial", 24)

    if won:
        title = font_big.render("YOU WIN!", True, YELLOW)
    else:
        title = font_big.render("GAME OVER", True, (255, 0, 0))

    score_text = font_small.render(f"Final Score: {score}", True, WHITE)
    restart_text = font_small.render("Press SPACE to restart, ESC to quit", True, WHITE)

    cx = width // 2
    cy = height // 2
    surface.blit(title, (cx - title.get_width() // 2, cy - 60))
    surface.blit(score_text, (cx - score_text.get_width() // 2, cy))
    surface.blit(restart_text, (cx - restart_text.get_width() // 2, cy + 40))


def main():
    pygame.init()

    game = PacManGame()
    state = game.reset()

    width = game.cols * TILE_SIZE
    height = game.rows * TILE_SIZE + SCORE_BAR_HEIGHT

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game.game_over:
                    state = game.reset()
                elif event.key == pygame.K_UP:
                    game.set_direction(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    game.set_direction(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    game.set_direction(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    game.set_direction(Direction.RIGHT)

        # Game tick
        if not game.game_over:
            state, reward, done = game.step()

        # Render
        screen.fill(BG_COLOR)

        draw_score(screen, state["score"], width,
                   state.get("invincible", False), state.get("invincible_timer", 0))
        draw_maze(screen, state["grid"], SCORE_BAR_HEIGHT)

        # Draw ghosts
        for i, (gr, gc, active, stunned, *_) in enumerate(state["ghosts"]):
            color = GHOST_COLORS[i] if i < len(GHOST_COLORS) else (255, 255, 255)
            draw_ghost(screen, gr, gc, color, active, stunned, state["tick"], SCORE_BAR_HEIGHT)

        # Draw Pac-Man
        pr, pc = state["pac_pos"]
        draw_pacman(screen, pr, pc, state["pac_dir"], state["tick"], SCORE_BAR_HEIGHT,
                    invincible=state.get("invincible", False))

        # Game over overlay
        if state["game_over"]:
            draw_game_over(screen, state["won"], state["score"], width, height)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
