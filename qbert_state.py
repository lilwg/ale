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

# Entity position addresses: slot 0 = Q*bert, slots 1-5 = enemies
# X positions: RAM[43..48], Y positions: RAM[67..72]
ENTITY_X_ADDRS = [43, 44, 45, 46, 47, 48]
ENTITY_Y_ADDRS = [67, 68, 69, 70, 71, 72]
NUM_ENTITY_SLOTS = 6

# RAM addresses for cube colors, indexed by (row, col)
# Verified: (1,0)→52, (2,1)→85, (4,1)→3, (5,1)→34
CUBE_RAM = {
    (0, 0): 21,
    (1, 0): 52,  (1, 1): 54,
    (2, 0): 83,  (2, 1): 85,  (2, 2): 87,
    (3, 0): 98,  (3, 1): 100, (3, 2): 102, (3, 3): 104,
    (4, 0): 1,   (4, 1): 3,   (4, 2): 5,   (4, 3): 7,   (4, 4): 9,
    (5, 0): 32,  (5, 1): 34,  (5, 2): 36,  (5, 3): 38,  (5, 4): 40,  (5, 5): 42,
}

# Pixel-to-grid constants
GRID_Y_BASE = 25
GRID_Y_STEP = 28
GRID_X_BASE = 77
GRID_X_ROW_SHIFT = -12
GRID_X_COL_STEP = 24

# Enemy colors for pixel detection
COILY_COLOR = np.array([146, 70, 192])
GREEN_COLOR = np.array([50, 132, 50])
RED_BALL_COLOR = np.array([200, 72, 72])

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
    __slots__ = ['qbert', 'coily', 'green', 'red_ball', 'enemies', 'lives', 'reward', 'done']

    def __init__(self):
        self.qbert = None    # (row, col) or None
        self.coily = None    # (row, col) or None
        self.green = None    # (row, col) or None
        self.red_ball = None # (row, col) or None
        self.enemies = []    # list of (row, col) for all detected enemies on the grid
        self.lives = 0
        self.reward = 0.0
        self.done = False

    def __repr__(self):
        parts = [f"Q={self.qbert}"]
        if self.coily:
            parts.append(f"C={self.coily}")
        if self.enemies:
            parts.append(f"E={self.enemies}")
        parts.append(f"lives={self.lives}")
        return f"State({', '.join(parts)})"


class QbertStateReader:
    """Reads game state from RAM + pixel fallback for enemies."""

    def __init__(self, env):
        self.env = env
        self._prev_y = None
        self._prev_x = None
        self._stable_count = 0
        self._cube_initial_color = None
        self._cube_start_values = None  # per-cube baseline at level start
        self._level = 1

    # Known level color cycle (verified empirically, repeats every 4 levels)
    LEVEL_COLORS = {1: 148, 2: 26, 3: 10, 4: 152}

    def set_level(self, level):
        """Set the current level and snapshot cube baseline for change detection."""
        self._level = level
        # Use known color cycle (period 4)
        cycle_pos = ((level - 1) % 4) + 1
        self._cube_initial_color = self.LEVEL_COLORS[cycle_pos]
        # Snapshot current cube values as the baseline for this level.
        # A cube is "done" when its value CHANGES from this baseline.
        # This handles carry-over from previous levels correctly.
        ram = self.env.unwrapped.ale.getRAM()
        self._cube_start_values = {(r, c): int(ram[addr])
                                    for (r, c), addr in CUBE_RAM.items()}

    def wait_for_level_start(self, first_action=5, max_frames=300):
        """Two-phase level start: NOOP during celebration, then first move.
        Phase 1: Send NOOP, wait for Q*bert to leave then return to (0,0).
        Phase 2: Send first_action until Q*bert moves. Capture baseline.
        Returns (obs, total_reward, done, info)."""
        total_r = 0
        obs = None
        info = {}
        done = False
        left_top = False
        at_top_count = 0
        # Phase 1: NOOP during celebration (don't risk DOWN during animation)
        for _ in range(max_frames):
            obs, r, t, tr, info = self.env.step(0)  # NOOP
            total_r += r
            if t or tr:
                done = True
                break
            pos = self.read_qbert_position()
            if pos == (0, 0):
                if left_top:
                    at_top_count += 1
                    if at_top_count >= 3:
                        break  # Celebration over, Q*bert stable at (0,0)
            else:
                left_top = True
                at_top_count = 0
        if done:
            return obs, total_r, done, info
        # Phase 2: send first_action until Q*bert moves (level started)
        for _ in range(60):
            # Snapshot baseline BEFORE each attempt
            ram = self.env.unwrapped.ale.getRAM()
            self._cube_start_values = {(r, c): int(ram[addr])
                                        for (r, c), addr in CUBE_RAM.items()}
            obs, r, t, tr, info = self.env.step(first_action)
            total_r += r
            if t or tr:
                done = True
                break
            pos = self.read_qbert_position()
            if pos is not None and pos != (0, 0):
                break  # Move worked — level started
        # Let Q*bert finish landing
        if not done:
            obs, r, done, info = self.wait_for_landing(15)
            total_r += r
        return obs, total_r, done, info

    def read_cube_done(self):
        """Read cube done/not-done state from RAM using per-cube baseline.
        A cube is 'done' when its value CHANGES from the level-start baseline.
        This handles carry-over from previous levels correctly."""
        ram = self.env.unwrapped.ale.getRAM()
        cube_done = [[False] * (r + 1) for r in range(MAX_ROW + 1)]
        baseline = self._cube_start_values
        if baseline is None:
            return cube_done
        for (r, c), addr in CUBE_RAM.items():
            cube_done[r][c] = (int(ram[addr]) != baseline[(r, c)])
        return cube_done

    def count_done_cubes(self):
        """Count completed cubes from RAM using per-cube baseline."""
        ram = self.env.unwrapped.ale.getRAM()
        baseline = self._cube_start_values
        if baseline is None:
            return 0
        return sum(1 for (r, c), addr in CUBE_RAM.items()
                   if int(ram[addr]) != baseline[(r, c)])

    def read_qbert_position(self):
        """Read Q*bert's grid position directly from RAM. Instant."""
        ram = self.env.unwrapped.ale.getRAM()
        y = int(ram[QBERT_Y_ADDR])
        x = int(ram[QBERT_X_ADDR])
        return ram_to_grid(y, x)

    def read_enemies_ram(self):
        """Read ALL enemy positions from RAM (slots 1-5, slot 0 is Q*bert).
        Returns list of (row, col) for each enemy on a valid grid position."""
        ram = self.env.unwrapped.ale.getRAM()
        qbert_pos = self.read_qbert_position()
        enemies = []
        for slot in range(1, NUM_ENTITY_SLOTS):
            y = int(ram[ENTITY_Y_ADDRS[slot]])
            x = int(ram[ENTITY_X_ADDRS[slot]])
            pos = ram_to_grid(y, x)
            if pos is not None and pos != qbert_pos:
                enemies.append(pos)
        return enemies

    def read_enemies(self, frame):
        """Read enemy positions from pixels. Returns (coily, green, red_ball)."""
        coily_px = find_sprite(frame, COILY_COLOR)
        green_px = find_sprite(frame, GREEN_COLOR)
        red_px = find_sprite(frame, RED_BALL_COLOR)
        coily = pixel_to_grid(coily_px[0], coily_px[1]) if coily_px else None
        green = pixel_to_grid(green_px[0], green_px[1]) if green_px else None
        red_ball = pixel_to_grid(red_px[0], red_px[1]) if red_px else None
        return coily, green, red_ball

    def read_state(self, obs, info, reward=0.0, done=False):
        """Read full game state. Returns QbertState."""
        state = QbertState()
        state.qbert = self.read_qbert_position()
        state.coily, state.green, state.red_ball = self.read_enemies(obs)
        state.enemies = self.read_enemies_ram()
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
