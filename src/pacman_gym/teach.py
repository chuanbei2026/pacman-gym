"""
RL Teaching Visualization: Watch DQN train live with neural network internals displayed.
Usage: python3 -m pacman_gym teach
"""

import copy
import math
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from .gym_env import PacManEnv
from .main import (
    draw_maze, draw_pacman, draw_ghost, draw_score,
    TILE_SIZE, SCORE_BAR_HEIGHT, GHOST_COLORS, BG_COLOR, WHITE, YELLOW, BLACK,
)
from .train import DQN, DQNAgent, MODEL_PATH

# Layout
GAME_WIDTH = 21 * TILE_SIZE   # 588
GAME_HEIGHT = 22 * TILE_SIZE + SCORE_BAR_HEIGHT  # 666
SLIDER_HEIGHT = 50
PANEL_WIDTH = 520
PANEL_MARGIN = 18
TOTAL_WIDTH = GAME_WIDTH + PANEL_WIDTH
TOTAL_HEIGHT = GAME_HEIGHT + SLIDER_HEIGHT
PANEL_X = GAME_WIDTH + PANEL_MARGIN

# Panel colors - academic / research style
PANEL_BG = (252, 252, 252)
TEXT_PRIMARY = (30, 30, 30)
TEXT_SECONDARY = (100, 100, 100)
TEXT_LABEL = (80, 80, 80)
ACCENT_BLUE = (31, 119, 180)     # matplotlib tab:blue
ACCENT_ORANGE = (255, 127, 14)   # matplotlib tab:orange
ACCENT_GREEN = (44, 160, 44)     # matplotlib tab:green
ACCENT_RED = (214, 39, 40)       # matplotlib tab:red
BORDER_COLOR = (200, 200, 200)
TABLE_HEADER_BG = (235, 240, 248)
TABLE_ROW_ALT = (245, 245, 250)
CHART_BG = (250, 250, 250)
CHART_GRID = (225, 225, 225)
SEPARATOR = (210, 210, 210)

# Speed slider
SPEED_MIN = 1
SPEED_MAX = 60
SLIDER_COLOR = (180, 180, 180)
SLIDER_KNOB = ACCENT_BLUE

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_COLORS = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED]

RENDER_FPS = 30          # UI frame rate
TRAIN_STEPS_PER_FRAME = 20 # run N game steps per rendered frame
DEATH_PAUSE_SEC = 0.5
GAME_FPS = 5             # original game speed for speed calculation


def get_q_values(net, obs, device):
    """Get Q-values for current state using a given network."""
    with torch.no_grad():
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        q = net(state_t)[0].cpu().numpy()
    return q


def select_action_with_net(net, obs, device, n_actions, epsilon):
    """Select action using a given network (epsilon-greedy)."""
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        return net(state_t).argmax(dim=1).item()


def get_weight_stats(agent):
    """Get weight magnitude heatmaps (downsampled) for each linear layer."""
    layers = []
    for module in agent.policy_net.net:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().cpu().numpy()
            layers.append(w)
    return layers


def weight_to_surface(w, max_w, max_h):
    """Convert a weight matrix to a pygame Surface as a heatmap."""
    rows, cols = w.shape
    # Downsample to fit
    step_r = max(1, rows // max_h)
    step_c = max(1, cols // max_w)
    sampled = w[::step_r, ::step_c]
    h, wi = sampled.shape
    h = min(h, max_h)
    wi = min(wi, max_w)
    sampled = sampled[:h, :wi]

    # Normalize to [-1, 1]
    vmax = max(abs(sampled.max()), abs(sampled.min()), 1e-6)
    normed = sampled / vmax

    surf = pygame.Surface((wi, h))
    pixels = pygame.PixelArray(surf)
    for r in range(h):
        for c in range(wi):
            v = normed[r, c]
            if v > 0:
                pixels[c, r] = (int(v * 200), int(v * 60), 0)
            else:
                pixels[c, r] = (0, int(-v * 60), int(-v * 200))
    del pixels
    return surf


def _draw_section_title(surface, text, x, y, font):
    """Draw a section title with underline."""
    rendered = font.render(text, True, TEXT_PRIMARY)
    surface.blit(rendered, (x, y))
    line_y = y + rendered.get_height() + 2
    pygame.draw.line(surface, SEPARATOR, (x, line_y), (x + PANEL_WIDTH - PANEL_MARGIN * 2, line_y), 1)
    return line_y + 8


def _draw_table(surface, headers, rows, x, y, col_widths, font_h, font_r):
    """Draw a table with header and alternating row colors."""
    row_h = 22
    # Header
    pygame.draw.rect(surface, TABLE_HEADER_BG, (x, y, sum(col_widths), row_h))
    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + sum(col_widths), y), 1)
    cx = x
    for i, h in enumerate(headers):
        surface.blit(font_h.render(h, True, TEXT_PRIMARY), (cx + 6, y + 4))
        cx += col_widths[i]
    y += row_h
    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + sum(col_widths), y), 1)

    # Rows
    for ri, row in enumerate(rows):
        bg = TABLE_ROW_ALT if ri % 2 == 1 else PANEL_BG
        pygame.draw.rect(surface, bg, (x, y, sum(col_widths), row_h))
        cx = x
        for i, val in enumerate(row):
            surface.blit(font_r.render(str(val), True, TEXT_SECONDARY), (cx + 6, y + 4))
            cx += col_widths[i]
        y += row_h

    # Bottom border
    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + sum(col_widths), y), 1)
    # Side borders
    total_w = sum(col_widths)
    top_y = y - row_h * (len(rows) + 1)
    pygame.draw.line(surface, BORDER_COLOR, (x, top_y), (x, y), 1)
    pygame.draw.line(surface, BORDER_COLOR, (x + total_w, top_y), (x + total_w, y), 1)
    return y + 6


def draw_panel(surface, agent, obs, q_values, chosen_action,
               episode, iteration, score, best_score, loss, loss_history,
               speed_mult=1.0, avg_score=0.0, survival_steps=0,
               scroll_offset=0):
    """Draw the right-side info panel in academic/research style.
    Returns the total content height for scrolling."""
    # Panel background - clean white
    panel_rect = pygame.Rect(GAME_WIDTH, 0, PANEL_WIDTH, TOTAL_HEIGHT)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)
    # Left border line
    pygame.draw.line(surface, BORDER_COLOR, (GAME_WIDTH, 0), (GAME_WIDTH, TOTAL_HEIGHT), 2)

    # Create a clip region for the panel
    surface.set_clip(pygame.Rect(GAME_WIDTH, 0, PANEL_WIDTH, TOTAL_HEIGHT))

    font_title = pygame.font.SysFont("arial", 16, bold=True)
    font_label = pygame.font.SysFont("arial", 13, bold=True)
    font_body = pygame.font.SysFont("arial", 13)
    font_small = pygame.font.SysFont("arial", 11)

    pw = PANEL_WIDTH - PANEL_MARGIN * 2  # usable panel width

    y = 12 - scroll_offset

    # --- Section: Q-Values (cross layout) ---
    y = _draw_section_title(surface, "Q-Values  (argmax policy)", PANEL_X, y, font_title)

    cross_cx = PANEL_X + pw // 2
    cross_cy = y + 70
    box_w, box_h = 110, 32
    gap = 8

    positions = {
        0: (cross_cx - box_w // 2, cross_cy - box_h - gap - box_h // 2),      # UP
        2: (cross_cx - box_w // 2, cross_cy + gap + box_h // 2),               # DOWN
        3: (cross_cx - box_w - gap - box_w // 2 + 35, cross_cy - box_h // 2),  # LEFT
        1: (cross_cx + gap + box_w // 2 - 35, cross_cy - box_h // 2),          # RIGHT
    }

    arrows = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}
    q_min = q_values.min()
    q_max = q_values.max()
    q_range = max(q_max - q_min, 1e-6)

    for i in range(4):
        bx, by = positions[i]
        is_chosen = (i == chosen_action)
        q_norm = (q_values[i] - q_min) / q_range  # 0..1

        # Box background
        pygame.draw.rect(surface, (235, 235, 235), (bx, by, box_w, box_h), border_radius=6)
        # Q-value magnitude fill
        fill_w = max(int(q_norm * box_w), 0)
        if fill_w > 0:
            fill_color = ACTION_COLORS[i]
            fill_surf = pygame.Surface((fill_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(fill_surf, (*fill_color, 140), (0, 0, fill_w, box_h), border_radius=6)
            surface.blit(fill_surf, (bx, by))
        # Border
        border_w = 3 if is_chosen else 1
        border_color = ACTION_COLORS[i] if is_chosen else BORDER_COLOR
        pygame.draw.rect(surface, border_color, (bx, by, box_w, box_h), border_w, border_radius=6)

        # Arrow + Q-value text
        arrow_txt = font_label.render(arrows[i], True, TEXT_PRIMARY)
        q_txt = font_body.render(f"{q_values[i]:.2f}", True, TEXT_PRIMARY)
        total_w = arrow_txt.get_width() + 6 + q_txt.get_width()
        start_x = bx + (box_w - total_w) // 2
        surface.blit(arrow_txt, (start_x, by + (box_h - arrow_txt.get_height()) // 2))
        surface.blit(q_txt, (start_x + arrow_txt.get_width() + 6, by + (box_h - q_txt.get_height()) // 2))

    y = cross_cy + box_h + gap + box_h // 2 + 12

    # --- Section: Weight Heatmaps ---
    y = _draw_section_title(surface, "Network Weights", PANEL_X, y, font_title)

    weights = get_weight_stats(agent)
    layer_names = ["Layer 1: Input -> 512", "Layer 2: 512 -> 256",
                   "Layer 3: 256 -> 64", "Layer 4: 64 -> Actions"]
    heatmap_h = 35
    heatmap_w = pw

    for w, name in zip(weights, layer_names):
        surface.blit(font_small.render(name, True, TEXT_SECONDARY), (PANEL_X, y))
        y += 14

        surf = weight_to_surface(w, heatmap_w, heatmap_h)
        scaled = pygame.transform.scale(surf, (heatmap_w, heatmap_h))
        surface.blit(scaled, (PANEL_X, y))
        pygame.draw.rect(surface, BORDER_COLOR, (PANEL_X, y, heatmap_w, heatmap_h), 1)
        y += heatmap_h + 5

    y += 4

    # --- Section: Training Loss Curve ---
    y = _draw_section_title(surface, "Training Loss", PANEL_X, y, font_title)

    curve_w = pw
    curve_h = 55
    # Chart background with grid
    pygame.draw.rect(surface, CHART_BG, (PANEL_X, y, curve_w, curve_h))
    for gy in range(4):
        ly = y + int(gy / 3 * (curve_h - 1))
        pygame.draw.line(surface, CHART_GRID, (PANEL_X, ly), (PANEL_X + curve_w, ly), 1)
    pygame.draw.rect(surface, BORDER_COLOR, (PANEL_X, y, curve_w, curve_h), 1)

    if len(loss_history) > 1:
        max_loss = max(loss_history) if max(loss_history) > 0 else 1
        points = []
        for i, l in enumerate(loss_history):
            px = PANEL_X + int(i / max(len(loss_history) - 1, 1) * (curve_w - 1))
            py = y + curve_h - 1 - int(min(l / max_loss, 1.0) * (curve_h - 2))
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(surface, ACCENT_BLUE, False, points, 2)

    y += curve_h + 10

    # --- Section: Training Stats Table ---
    y = _draw_section_title(surface, "Training Statistics", PANEL_X, y, font_title)

    table_rows = [
        ("Episode (session)", str(episode)),
        ("Episode (total)", str(agent.total_episodes)),
        ("Iteration", str(iteration)),
        ("Score", str(score)),
        ("Avg Score", f"{avg_score:.0f}"),
        ("Best Score", str(best_score)),
        ("Survival", f"{survival_steps} steps"),
        ("Loss", f"{loss:.4f}" if loss > 0 else "--"),
        ("Epsilon", f"{agent.epsilon:.3f}"),
        ("Buffer Size", str(len(agent.buffer))),
        ("Speed", f"{speed_mult:.0f}x realtime"),
    ]
    col1 = pw // 2
    col2 = pw - col1
    y = _draw_table(
        surface,
        headers=["Parameter", "Value"],
        rows=table_rows,
        x=PANEL_X, y=y,
        col_widths=[col1, col2],
        font_h=font_label, font_r=font_body,
    )

    content_height = y + scroll_offset + 10  # total content height
    surface.set_clip(None)

    # Draw scrollbar if content overflows
    if content_height > TOTAL_HEIGHT:
        sb_x = GAME_WIDTH + PANEL_WIDTH - 8
        sb_h = TOTAL_HEIGHT
        thumb_ratio = TOTAL_HEIGHT / content_height
        thumb_h = max(int(sb_h * thumb_ratio), 20)
        thumb_y = int(scroll_offset / max(content_height - TOTAL_HEIGHT, 1) * (sb_h - thumb_h))
        pygame.draw.rect(surface, (230, 230, 230), (sb_x, 0, 6, sb_h))
        pygame.draw.rect(surface, (160, 160, 170), (sb_x, thumb_y, 6, thumb_h), border_radius=3)

    return content_height


def draw_speed_slider(surface, x, y, w, h, steps_per_frame):
    """Draw the speed control slider below the game. Returns slider track rect."""
    font = pygame.font.SysFont("arial", 13, bold=True)
    font_val = pygame.font.SysFont("arial", 12)

    # Background
    pygame.draw.rect(surface, (20, 20, 20), (x, y, w, h))

    label = font.render("Speed:", True, (200, 200, 200))
    surface.blit(label, (x + 10, y + (h - label.get_height()) // 2))

    # Track
    track_x = x + 75
    track_w = w - 160
    track_y = y + h // 2
    pygame.draw.line(surface, SLIDER_COLOR, (track_x, track_y), (track_x + track_w, track_y), 3)

    # Knob position
    ratio = (steps_per_frame - SPEED_MIN) / max(SPEED_MAX - SPEED_MIN, 1)
    knob_x = track_x + int(ratio * track_w)
    pygame.draw.circle(surface, SLIDER_KNOB, (knob_x, track_y), 8)
    pygame.draw.circle(surface, (255, 255, 255), (knob_x, track_y), 5)

    # Speed text
    speed_mult = steps_per_frame * RENDER_FPS / GAME_FPS
    val_txt = font_val.render(f"{speed_mult:.0f}x", True, (200, 200, 200))
    surface.blit(val_txt, (track_x + track_w + 12, y + (h - val_txt.get_height()) // 2))

    return pygame.Rect(track_x, track_y - 10, track_w, 20)


def _render_game(screen, game_state):
    """Render the game side (left)."""
    draw_score(screen, game_state["score"], GAME_WIDTH,
               game_state.get("invincible", False),
               game_state.get("invincible_timer", 0))
    draw_maze(screen, game_state["grid"], SCORE_BAR_HEIGHT)
    for i, (gr, gc, active, stunned, *_) in enumerate(game_state["ghosts"]):
        color = GHOST_COLORS[i] if i < len(GHOST_COLORS) else (255, 255, 255)
        draw_ghost(screen, gr, gc, color, active, stunned,
                   game_state["tick"], SCORE_BAR_HEIGHT)
    pr, pc = game_state["pac_pos"]
    draw_pacman(screen, pr, pc, game_state["pac_dir"],
                game_state["tick"], SCORE_BAR_HEIGHT,
                invincible=game_state.get("invincible", False))


def teach():
    """Live training visualization mode with async training."""
    pygame.init()
    screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
    pygame.display.set_caption("Pac-Man RL Training Visualization")
    clock = pygame.time.Clock()

    env = PacManEnv(max_steps=2000)
    obs, _ = env.reset()

    agent = DQNAgent(obs.shape[0], env.action_space.n)
    if MODEL_PATH.exists():
        agent.load()

    # Rollout net: frozen copy for action selection (main thread)
    # policy_net is trained by background thread
    rollout_net = copy.deepcopy(agent.policy_net)
    rollout_net.eval()

    # --- Background training thread ---
    train_stats = {"loss": 0.0, "iterations": 0, "running": True}

    def train_loop():
        while train_stats["running"]:
            loss = agent.train_step()
            if loss > 0:
                train_stats["loss"] = loss
                train_stats["iterations"] += 1
            else:
                time.sleep(0.002)  # buffer not ready yet

    train_thread = threading.Thread(target=train_loop, daemon=True)
    train_thread.start()

    # --- State ---
    episode = 0
    best_score = 0
    loss_history = deque(maxlen=200)
    recent_scores = deque(maxlen=50)
    steps_per_frame = TRAIN_STEPS_PER_FRAME
    scroll_offset = 0
    panel_content_h = TOTAL_HEIGHT
    dragging_slider = False
    slider_track = pygame.Rect(0, 0, 0, 0)
    prev_iter = 0

    def sync_rollout_net():
        """Copy latest trained weights to rollout net."""
        rollout_net.load_state_dict(agent.policy_net.state_dict())
        rollout_net.eval()

    def process_events():
        """Handle events. Returns True if should quit."""
        nonlocal scroll_offset, steps_per_frame, dragging_slider, slider_track
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True

            if event.type == pygame.MOUSEWHEEL:
                mx, _ = pygame.mouse.get_pos()
                if mx > GAME_WIDTH:
                    scroll_offset -= event.y * 25
                    max_scroll = max(0, panel_content_h - TOTAL_HEIGHT)
                    scroll_offset = max(0, min(scroll_offset, max_scroll))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if slider_track.collidepoint(event.pos):
                    dragging_slider = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_slider = False

        if dragging_slider:
            mx, _ = pygame.mouse.get_pos()
            ratio = (mx - slider_track.x) / max(slider_track.w, 1)
            ratio = max(0.0, min(1.0, ratio))
            steps_per_frame = int(SPEED_MIN + ratio * (SPEED_MAX - SPEED_MIN))
            steps_per_frame = max(SPEED_MIN, min(SPEED_MAX, steps_per_frame))

        return False

    def render_frame(game_state, q, action, ep_score, avg_sc, ep_st,
                     cur_loss, cur_iter, overlay_text=None):
        nonlocal panel_content_h, slider_track
        screen.fill(BG_COLOR)

        _render_game(screen, game_state)

        speed_mult = steps_per_frame * RENDER_FPS / GAME_FPS
        slider_track = draw_speed_slider(
            screen, 0, GAME_HEIGHT, GAME_WIDTH, SLIDER_HEIGHT, steps_per_frame)

        if overlay_text:
            overlay = pygame.Surface((GAME_WIDTH, 50), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            oy = GAME_HEIGHT // 2 - 25
            screen.blit(overlay, (0, oy))
            font_big = pygame.font.SysFont("arial", 24, bold=True)
            txt = font_big.render(overlay_text, True, (255, 255, 255))
            screen.blit(txt, ((GAME_WIDTH - txt.get_width()) // 2, oy + 12))

        panel_content_h = draw_panel(
            screen, agent, obs, q, action,
            episode, cur_iter, ep_score, best_score, cur_loss, loss_history,
            speed_mult=speed_mult, avg_score=avg_sc,
            survival_steps=ep_st, scroll_offset=scroll_offset)

        pygame.display.flip()
        clock.tick(RENDER_FPS)

    running = True
    while running:
        obs, _ = env.reset()
        episode += 1
        agent.total_episodes += 1
        done = False
        ep_score = 0
        ep_steps = 0
        last_action = 0
        last_q = np.zeros(4)
        avg_score = np.mean(recent_scores) if recent_scores else 0.0

        # Sync rollout net at start of each episode
        sync_rollout_net()

        while not done and running:
            if process_events():
                running = False
                break

            # Rollout uses frozen rollout_net; training happens in background
            for _ in range(steps_per_frame):
                last_q = get_q_values(rollout_net, obs, agent.device)
                last_action = select_action_with_net(
                    rollout_net, obs, agent.device,
                    agent.n_actions, agent.epsilon)
                agent.steps_done += 1

                next_obs, reward, done, truncated, info = env.step(last_action)
                agent.buffer.push(obs, last_action, reward, next_obs, done or truncated)

                ep_score = info["score"]
                ep_steps = info["steps"]
                obs = next_obs

                if done or truncated:
                    done = True
                    break

            # Collect training stats from background thread
            cur_loss = train_stats["loss"]
            cur_iter = train_stats["iterations"]
            if cur_iter > prev_iter:
                loss_history.append(cur_loss)
                prev_iter = cur_iter

            game_state = env._game._get_state()
            render_frame(game_state, last_q, last_action,
                         ep_score, avg_score, ep_steps, cur_loss, cur_iter)

        # Episode ended
        recent_scores.append(ep_score)
        avg_score = np.mean(recent_scores)
        if ep_score > best_score:
            best_score = ep_score
            agent.save()
        elif episode % 50 == 0:
            agent.save()

        if not running:
            break

        # Brief pause between episodes (no blocking training needed - it's async)
        pause_frames = int(DEATH_PAUSE_SEC * RENDER_FPS)
        for f in range(pause_frames):
            if process_events():
                running = False
                break
            cur_loss = train_stats["loss"]
            cur_iter = train_stats["iterations"]
            if cur_iter > prev_iter:
                loss_history.append(cur_loss)
                prev_iter = cur_iter
            q_values = get_q_values(rollout_net, obs, agent.device)
            render_frame(game_state, q_values, -1,
                         ep_score, avg_score, ep_steps, cur_loss, cur_iter,
                         overlay_text="Syncing weights...")

    # Shutdown
    train_stats["running"] = False
    train_thread.join(timeout=2)
    pygame.quit()


if __name__ == "__main__":
    teach()
