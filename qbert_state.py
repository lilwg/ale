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
RED_BALL_COLOR = np.array([181, 83, 40])

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
        self._cube_target_color = None
        self._reward_done = None  # set of (r,c) confirmed done by reward signal
        self._level = 1

    # Known level color cycle (verified empirically, repeats every 4 levels)
    LEVEL_COLORS = {1: 148, 2: 26, 3: 10, 4: 152}

    def set_level(self, level):
        """Set the current level. Target color learned from first 25-reward."""
        self._level = level
        self._cube_target_color = None  # learned from first colored cube
        self._cube_start_values = None  # set by wait_for_level_start
        self._reward_done = None

    def enable_reward_tracking(self):
        """Switch to reward-based cube tracking (when baseline is wrong).
        Seeds with cubes currently marked as done by baseline tracking."""
        if self._reward_done is None:
            self._reward_done = set()
            # Seed from baseline tracking
            if self._cube_start_values is not None:
                ram = self.env.unwrapped.ale.getRAM()
                for (r, c), addr in CUBE_RAM.items():
                    if int(ram[addr]) != self._cube_start_values[(r, c)]:
                        self._reward_done.add((r, c))

    def mark_cube_done_by_reward(self, pos):
        """Mark a cube as done because Q*bert got a reward there."""
        if self._reward_done is not None and pos:
            self._reward_done.add(pos)

    def wait_for_level_start(self, max_frames=300):
        """Play a fixed safe pattern through celebration until a move gives reward.
        Pattern: 5 DL, 1 UR, (1 DR, 1 UL)*4, 1 DR — covers left column.
        During celebration the game ignores input. When level starts,
        first successful move gives 25 reward.
        Returns (obs, total_reward, done, info)."""
        total_r = 0
        obs = None
        info = {}
        done = False
        # DL=5, UR=2, DR=3, UL=4
        # Restart pattern each cycle so first real move is always DL from (0,0)
        pattern = [5]*5 + [2] + [3,4]*4 + [3]
        for i in range(max_frames):
            action = pattern[i % len(pattern)]
            obs, r, t, tr, info = self.env.step(action)
            total_r += r
            if t or tr:
                done = True
                break
            if r == 25:
                obs2, r2, done2, info = self.wait_for_landing(15)
                total_r += r2
                done = done2
                break
        if not done:
            from collections import Counter
            ram = self.env.unwrapped.ale.getRAM()
            vals = [int(ram[addr]) for addr in CUBE_RAM.values()]
            majority = Counter(vals).most_common(1)[0][0]
            self._cube_start_values = {rc: majority for rc in CUBE_RAM.keys()}
        return obs, total_r, done, info


    def learn_target_color(self, qbert_pos):
        """Learn the target color from a 25-point reward cube."""
        if qbert_pos and self._cube_target_color is None:
            r, c = qbert_pos
            addr = CUBE_RAM.get((r, c))
            if addr is not None:
                ram = self.env.unwrapped.ale.getRAM()
                self._cube_target_color = int(ram[addr])
                print(f"  ** Learned target color: {self._cube_target_color} at ({r},{c})")

    def read_cube_done(self):
        """Read cube done state from RAM.
        If target known: done = (value == target).
        Otherwise: done = (value != initial) using baseline."""
        cube_done = [[False] * (r + 1) for r in range(MAX_ROW + 1)]
        ram = self.env.unwrapped.ale.getRAM()
        if self._cube_target_color is not None:
            for (r, c), addr in CUBE_RAM.items():
                cube_done[r][c] = (int(ram[addr]) == self._cube_target_color)
        elif self._cube_start_values is not None:
            for (r, c), addr in CUBE_RAM.items():
                cube_done[r][c] = (int(ram[addr]) != self._cube_start_values[(r, c)])
        return cube_done

    def count_done_cubes(self):
        """Count completed cubes."""
        ram = self.env.unwrapped.ale.getRAM()
        if self._cube_target_color is not None:
            return sum(1 for addr in CUBE_RAM.values()
                       if int(ram[addr]) == self._cube_target_color)
        elif self._cube_start_values is not None:
            return sum(1 for (r, c), addr in CUBE_RAM.items()
                       if int(ram[addr]) != self._cube_start_values[(r, c)])
        return 0

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
        coily = pixel_to_grid(coily_px[0], coily_px[1]) if coily_px else None
        green = pixel_to_grid(green_px[0], green_px[1]) if green_px else None
        # Red ball: search at each grid position (excluding Q*bert) for the color
        red_ball = None
        qbert_pos = self.read_qbert_position()
        for (r, c), (gy, gx) in GRID_PIXELS.items():
            if (r, c) == qbert_pos:
                continue
            # Check a wider region around the grid center
            y0, y1 = max(0, gy - 15), min(frame.shape[0], gy + 15)
            x0, x1 = max(0, gx - 15), min(frame.shape[1], gx + 15)
            region = frame[y0:y1, x0:x1]
            diff = np.abs(region.astype(int) - RED_BALL_COLOR)
            matches = np.sum(np.all(diff < 25, axis=2))
            if matches >= 3:
                red_ball = (r, c)
                break  # found one, that's enough
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
