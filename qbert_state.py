"""
Q*bert RAM-based game state reader for Atari 2600 (ALE/Gymnasium).

Reads Q*bert's grid position directly from RAM — no pixel scanning needed.
Detects jump completion by polling RAM for position stability.
Uses pixel color matching as fallback for enemy (Coily/green ball) detection.

Usage:
    import gymnasium as gym
    import ale_py
    from qbert_state import QbertStateReader

    env = gym.make("ALE/Qbert-v5", render_mode="human")
    reader = QbertStateReader(env)
    obs, info = env.reset()
    obs, _, done, info = reader.wait_for_game_start()
    state = reader.read_state(obs, info)
    print(state)  # State(Q=(0, 0), lives=3)
"""

import numpy as np

MAX_ROW = 5
NUM_CUBES = 21

# RAM addresses (Atari 2600 Q*bert, verified empirically)
QBERT_Y_ADDR = 67   # Q*bert's Y pixel position
QBERT_X_ADDR = 43   # Q*bert's X pixel position

# Pixel-to-grid constants
GRID_Y_BASE = 25
GRID_Y_STEP = 28
GRID_X_BASE = 78
GRID_X_ROW_SHIFT = -13
GRID_X_COL_STEP = 24

# Enemy colors for pixel detection
COILY_COLOR = np.array([146, 70, 192])
GREEN_COLOR = np.array([50, 132, 50])

# Grid pixel lookup (for enemy detection)
GRID_PIXELS = {}
for r in range(6):
    for c in range(r + 1):
        GRID_PIXELS[(r, c)] = (
            GRID_Y_BASE + r * GRID_Y_STEP,
            GRID_X_BASE + r * GRID_X_ROW_SHIFT + c * GRID_X_COL_STEP,
        )


def is_valid(row, col):
    return 0 <= row <= MAX_ROW and 0 <= col <= row


def ram_to_grid(y_val, x_val):
    """Convert RAM Y/X values to grid (row, col). Returns None if invalid."""
    row = round((y_val - GRID_Y_BASE) / GRID_Y_STEP)
    if row < 0 or row > MAX_ROW:
        return None
    col = round((x_val - GRID_X_BASE - row * GRID_X_ROW_SHIFT) / GRID_X_COL_STEP)
    if col < 0 or col > row:
        return None
    # Verify it's close enough to a real grid position
    expected_y = GRID_Y_BASE + row * GRID_Y_STEP
    expected_x = GRID_X_BASE + row * GRID_X_ROW_SHIFT + col * GRID_X_COL_STEP
    if abs(y_val - expected_y) > 8 or abs(x_val - expected_x) > 8:
        return None
    return (row, col)


def pixel_to_grid(py, px):
    """Map pixel position to nearest grid cell."""
    best = None
    best_dist = 999
    for (r, c), (gy, gx) in GRID_PIXELS.items():
        d = ((py - gy) ** 2 + (px - gx) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = (r, c)
    return best if best_dist <= 25 else None


def find_sprite(frame, color, tol=40, min_pixels=3):
    """Find a sprite by color. Returns (y, x) or None."""
    diff = np.abs(frame.astype(int) - color)
    mask = np.all(diff < tol, axis=2)
    ys, xs = np.where(mask)
    if len(ys) < min_pixels:
        return None
    return (int(np.mean(ys)), int(np.mean(xs)))


class QbertState:
    """Snapshot of the game state."""
    __slots__ = ['qbert', 'coily', 'green', 'lives', 'reward', 'done']

    def __init__(self):
        self.qbert = None    # (row, col) or None
        self.coily = None    # (row, col) or None
        self.green = None    # (row, col) or None
        self.lives = 0
        self.reward = 0.0
        self.done = False

    def __repr__(self):
        parts = [f"Q={self.qbert}"]
        if self.coily:
            parts.append(f"C={self.coily}")
        if self.green:
            parts.append(f"G={self.green}")
        parts.append(f"lives={self.lives}")
        return f"State({', '.join(parts)})"


class QbertStateReader:
    """Reads game state from RAM + pixel fallback for enemies."""

    def __init__(self, env):
        self.env = env
        self._prev_y = None
        self._prev_x = None
        self._stable_count = 0

    def read_qbert_position(self):
        """Read Q*bert's grid position directly from RAM. Instant."""
        ram = self.env.unwrapped.ale.getRAM()
        y = int(ram[QBERT_Y_ADDR])
        x = int(ram[QBERT_X_ADDR])
        return ram_to_grid(y, x)

    def read_enemies(self, frame):
        """Read enemy positions from pixels. Returns (coily, green)."""
        coily_px = find_sprite(frame, COILY_COLOR)
        green_px = find_sprite(frame, GREEN_COLOR)
        coily = pixel_to_grid(coily_px[0], coily_px[1]) if coily_px else None
        green = pixel_to_grid(green_px[0], green_px[1]) if green_px else None
        return coily, green

    def read_state(self, obs, info, reward=0.0, done=False):
        """Read full game state. Returns QbertState."""
        state = QbertState()
        state.qbert = self.read_qbert_position()
        state.coily, state.green = self.read_enemies(obs)
        state.lives = info.get('lives', 0)
        state.reward = reward
        state.done = done
        return state

    def wait_for_landing(self, max_frames=20):
        """Wait for Q*bert to land by polling RAM for position stability.
        Much faster than pixel-based wait_stable.
        Returns (obs, total_reward, done, info)."""
        total_r = 0
        obs = None
        info = {}
        done = False
        prev_y = None
        prev_x = None
        stable = 0

        for _ in range(max_frames):
            obs, r, t, tr, info = self.env.step(0)
            total_r += r
            if t or tr:
                done = True
                break

            ram = self.env.unwrapped.ale.getRAM()
            y = int(ram[QBERT_Y_ADDR])
            x = int(ram[QBERT_X_ADDR])

            # Check if position is stable AND maps to a valid grid cell
            if prev_y is not None and y == prev_y and x == prev_x:
                grid = ram_to_grid(y, x)
                if grid is not None:
                    stable += 1
            else:
                stable = 0

            prev_y = y
            prev_x = x

            if stable >= 2:  # Much faster than pixel-based (needed 5)
                break

        return obs, total_r, done, info

    def wait_for_game_start(self):
        """Wait for Q*bert to appear on the pyramid at game start."""
        total_r = 0
        obs = None
        info = {}

        # Initial wait
        for _ in range(80):
            obs, r, t, tr, info = self.env.step(0)
            total_r += r
            if t or tr:
                return obs, total_r, True, info

        # Wait for Q*bert to settle
        obs, r, done, info = self.wait_for_landing(max_frames=40)
        total_r += r
        return obs, total_r, done, info
