"""
Q*bert Heuristic Agent — RAM-based state reading + disc strategy.

BFS-based pathfinding to unvisited cubes, Coily avoidance with predicted
chase movement, and disc lure kills. Uses graph-peeling routing on toggle
levels (3+) to complete cubes outside-in and avoid reverting progress.
Clears 5+ levels (~14,500 score).

Usage:
    python qbert_heuristic.py          # Watch the agent play (renders game window)

Requires: gymnasium, ale-py, numpy
"""

import heapq
import sys
import gymnasium as gym
import ale_py
import numpy as np
from collections import deque, Counter

from qbert_state import (
    QbertStateReader, is_valid, MAX_ROW, NUM_CUBES, CUBE_RAM,
)

gym.register_envs(ale_py)

NOOP = 0
UP = 2      # row-1, col    (up-right)
RIGHT = 3   # row+1, col+1  (down-right)
LEFT = 4    # row-1, col-1  (up-left)
DOWN = 5    # row+1, col    (down-left)

MOVES = {
    DOWN:  (+1,  0, "down-left"),
    RIGHT: (+1, +1, "down-right"),
    UP:    (-1,  0, "up-right"),
    LEFT:  (-1, -1, "up-left"),
}

DISCS = {(4, 0): LEFT, (4, 4): UP}


def neighbors(row, col):
    result = []
    for action, (dr, dc, _) in MOVES.items():
        nr, nc = row + dr, col + dc
        if is_valid(nr, nc):
            result.append((action, nr, nc))
    return result


# --- Peel layers (precomputed for fixed pyramid topology) ---
# Graph peeling iteratively removes minimum-degree nodes, assigning layers.
# Outer/corner nodes get low layers, inner nodes get high layers.
# Used on toggle levels (3+) to complete cubes outside-in.

def _compute_peel_layers():
    all_pos = [(r, c) for r in range(MAX_ROW + 1) for c in range(r + 1)]
    adj = {p: [(nr, nc) for _, nr, nc in neighbors(p[0], p[1])] for p in all_pos}
    remaining = set(all_pos)
    layers = {}
    layer = 0
    while remaining:
        min_deg = min(len([n for n in adj[p] if n in remaining]) for p in remaining)
        batch = [p for p in remaining
                 if len([n for n in adj[p] if n in remaining]) == min_deg]
        for p in batch:
            layers[p] = layer
            remaining.discard(p)
        layer += 1
    return layers

PEEL_LAYERS = _compute_peel_layers()
MAX_PEEL_LAYER = max(PEEL_LAYERS.values())


def grid_distance(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def predict_coily_move(cr, cc, qr, qc):
    """Predict Coily's best chase move. Returns single (row, col)."""
    best = (cr, cc)
    best_dist = grid_distance(cr, cc, qr, qc)
    for _, nr, nc in neighbors(cr, cc):
        d = grid_distance(nr, nc, qr, qc)
        if d < best_dist:
            best_dist = d
            best = (nr, nc)
    return best


def predict_coily_moves(cr, cc, qr, qc):
    """Return ALL positions Coily could move to (handles ties).
    Only includes moves that strictly improve distance (matching predict_coily_move).
    Returns [(cr,cc)] if no improving move exists (Coily stays)."""
    curr_dist = grid_distance(cr, cc, qr, qc)
    best_dist = curr_dist
    for _, nr, nc in neighbors(cr, cc):
        d = grid_distance(nr, nc, qr, qc)
        if d < best_dist:
            best_dist = d
    if best_dist >= curr_dist:
        return [(cr, cc)]  # No improvement — Coily stays
    return [(nr, nc) for _, nr, nc in neighbors(cr, cc)
            if grid_distance(nr, nc, qr, qc) == best_dist]


def simulate_coily(coily, qbert_path):
    """Simulate Coily chasing Q*bert over multiple steps.
    qbert_path is a list of (row, col) positions Q*bert will visit.
    Returns list of Coily positions after each step."""
    cr, cc = coily
    positions = []
    for qr, qc in qbert_path:
        cr, cc = predict_coily_move(cr, cc, qr, qc)
        positions.append((cr, cc))
    return positions


def count_escape_routes(row, col, coily):
    """Count how many safe moves exist from a position (considering Coily)."""
    count = 0
    for _, nr, nc in neighbors(row, col):
        if coily and (nr, nc) == coily:
            continue
        if coily:
            pcr, pcc = predict_coily_move(coily[0], coily[1], nr, nc)
            if (pcr, pcc) == (nr, nc):
                continue
        count += 1
    return count


def survives_n_steps(row, col, coily, n=3):
    """Check if Q*bert can survive n steps from (row, col) with Coily chasing.
    Returns True if at least one safe path of length n exists."""
    if n <= 0:
        return True
    if coily is None:
        return True
    for _, nr, nc in neighbors(row, col):
        # Check if this move is safe (Coily won't land on us)
        if (nr, nc) == coily:
            continue
        possible = predict_coily_moves(coily[0], coily[1], nr, nc)
        if (nr, nc) in possible:
            continue
        # Simulate Coily's move and recurse
        pc = predict_coily_move(coily[0], coily[1], nr, nc)
        if survives_n_steps(nr, nc, pc, n - 1):
            return True
    return False


def coily_can_reach(coily, target):
    """Check if Coily is on target or adjacent to it (can reach in 1 step)."""
    if coily == target:
        return True
    for _, nr, nc in neighbors(coily[0], coily[1]):
        if (nr, nc) == target:
            return True
    return False


def is_move_safe(row, col, action, coily):
    dr, dc, _ = MOVES[action]
    nr, nc = row + dr, col + dc
    if not is_valid(nr, nc):
        return nr, nc, coily, False
    if coily is None:
        return nr, nc, None, True
    # Unsafe if Coily is on the destination
    if (coily[0], coily[1]) == (nr, nc):
        return nr, nc, coily, False
    # Check Coily's simultaneous move (step 1) and next move (step 2)
    step1_moves = predict_coily_moves(coily[0], coily[1], nr, nc)
    if (nr, nc) in step1_moves:
        return nr, nc, coily, False
    for s1r, s1c in step1_moves:
        step2_moves = predict_coily_moves(s1r, s1c, nr, nc)
        if (nr, nc) in step2_moves:
            return nr, nc, coily, False
    pcr, pcc = predict_coily_move(coily[0], coily[1], nr, nc)
    return nr, nc, (pcr, pcc), True


def has_safe_followup(row, col, action, coily):
    """After taking action, does at least one safe 2nd move exist?"""
    nr, nc, predicted_coily, safe = is_move_safe(row, col, action, coily)
    if not safe or not is_valid(nr, nc):
        return False
    for _, nnr, nnc in neighbors(nr, nc):
        if predicted_coily is None:
            return True
        if (predicted_coily[0], predicted_coily[1]) == (nnr, nnc):
            continue
        pcr, pcc = predict_coily_move(predicted_coily[0], predicted_coily[1], nnr, nnc)
        if (pcr, pcc) != (nnr, nnc):
            return True
    return False


def bfs_path_to(sr, sc, tr, tc, blocked=set()):
    if (sr, sc) == (tr, tc):
        return None
    seen = {(sr, sc)}
    queue = deque()
    for action, nr, nc in neighbors(sr, sc):
        if (nr, nc) not in seen and (nr, nc) not in blocked:
            if (nr, nc) == (tr, tc):
                return action
            seen.add((nr, nc))
            queue.append((nr, nc, action))
    while queue:
        cr, cc, first_action = queue.popleft()
        for _, nr, nc in neighbors(cr, cc):
            if (nr, nc) not in seen and (nr, nc) not in blocked:
                if (nr, nc) == (tr, tc):
                    return first_action
                seen.add((nr, nc))
                queue.append((nr, nc, first_action))
    return None


def bfs_nearest_undone(row, col, cube_done, blocked=set()):
    """BFS to nearest cube where cube_done[r][c] is False."""
    seen = {(row, col)}
    queue = deque()
    for action, nr, nc in neighbors(row, col):
        if (nr, nc) not in seen:
            # Allow reaching blocked positions as DESTINATIONS (undone cubes)
            # but don't traverse THROUGH blocked positions
            if not cube_done[nr][nc]:
                return action
            if (nr, nc) not in blocked:
                seen.add((nr, nc))
                queue.append((nr, nc, action))
    while queue:
        cr, cc, first_action = queue.popleft()
        for _, nr, nc in neighbors(cr, cc):
            if (nr, nc) not in seen:
                if not cube_done[nr][nc]:
                    return first_action
                if (nr, nc) not in blocked:
                    seen.add((nr, nc))
                    queue.append((nr, nc, first_action))
    return None


def bfs_peel_route(row, col, cube_done, blocked=set(), reversion_penalty=4):
    """Dijkstra toward nearest undone cube in lowest incomplete peel layer.
    Adds +4 cost for traversing done cubes (reversion penalty on toggle levels)."""
    # Find target: lowest peel layer with any undone cubes
    target_layer = None
    for layer in range(MAX_PEEL_LAYER + 1):
        for pos, pl in PEEL_LAYERS.items():
            if pl == layer and not cube_done[pos[0]][pos[1]]:
                target_layer = layer
                break
        if target_layer is not None:
            break
    if target_layer is None:
        return None

    # Dijkstra with reversion penalty
    dist = {(row, col): 0}
    heap = [(0, row, col, None)]
    while heap:
        cost, cr, cc, first_action = heapq.heappop(heap)
        if cost > dist.get((cr, cc), float('inf')):
            continue
        if first_action is not None \
                and PEEL_LAYERS[(cr, cc)] == target_layer \
                and not cube_done[cr][cc]:
            return first_action
        for action, nr, nc in neighbors(cr, cc):
            # Allow reaching blocked undone cubes as destinations
            if (nr, nc) in blocked and cube_done[nr][nc]:
                continue
            step_cost = 1
            if cube_done[nr][nc]:
                step_cost += reversion_penalty
            if (nr, nc) in blocked:
                step_cost += 8  # high cost but not impossible
            new_cost = cost + step_cost
            if new_cost < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = new_cost
                fa = first_action if first_action is not None else action
                heapq.heappush(heap, (new_cost, nr, nc, fa))
    return None


_no_progress_count = 0
_route_target = None
_on_second_pass = False
_prev_level_was_multi_hit = False

def pick_action(row, col, cube_done, state, discs_available, level=1):
    global _no_progress_count, _route_target
    coily = state.coily

    # Danger zone: Coily only (pixel-detected, reliable).
    # RAM entity slots have false positives that cause phantom danger zones.
    danger = set()
    if coily:
        danger.add(coily)
        for _, nr, nc in neighbors(coily[0], coily[1]):
            danger.add((nr, nc))

    coily_dist = grid_distance(row, col, coily[0], coily[1]) if coily else 99

    # Pre-compute safe moves with 3-step lookahead
    valid_moves = []
    for action, nr, nc in neighbors(row, col):
        _, _, predicted_coily, safe = is_move_safe(row, col, action, coily)
        followup = has_safe_followup(row, col, action, coily) if safe else False
        escapes = count_escape_routes(nr, nc, predicted_coily if safe else coily)
        # 3-step survival check: can Q*bert survive 3 more steps from this position?
        survives = survives_n_steps(nr, nc, predicted_coily, 3) if safe else False
        valid_moves.append((action, nr, nc, safe, predicted_coily, followup, escapes, survives))
    safe_moves = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, _, esc, surv in valid_moves if s and surv]
    safe_with_followup = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, f, esc, surv in valid_moves if s and f and surv]
    # Fallback if 3-step filter is too strict
    if not safe_moves:
        safe_moves = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, _, esc, _ in valid_moves if s]
    if not safe_with_followup:
        safe_with_followup = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, f, esc, _ in valid_moves if s and f]

    # Count remaining cubes
    cubes_remaining = sum(
        1 for r in range(MAX_ROW + 1) for c in range(r + 1) if not cube_done[r][c]
    )


    # FINISH LEVEL: if few cubes left, rush to complete.
    # When no discs, be more aggressive (fleeing just delays death).
    finish_threshold = 5 if not discs_available else 3
    if cubes_remaining <= finish_threshold:
        action = bfs_nearest_undone(row, col, cube_done)
        if action is not None:
            nr, nc, pc, safe = is_move_safe(row, col, action, coily)
            if safe and (not coily or survives_n_steps(nr, nc, pc, 3)):
                return action

    # --- DISC STRATEGY ---
    # Only use discs against Coily (snake form, actively chasing).
    # The ball form bounces from the top — don't waste discs on it.
    # Heuristic: Coily (snake) is typically at row >= 3 when chasing.
    coily_is_snake = coily and coily[0] >= 3
    if coily_is_snake and discs_available:
        on_disc = (row, col) in discs_available

        # Fire disc when Coily is adjacent (guaranteed kill)
        if on_disc and coily_dist <= 1:
            return DISCS[(row, col)]

        # Wait on disc for Coily to approach — don't leave!
        if on_disc and coily_dist <= 4:
            # Stay near the disc: pick a neighbor that's closest to the disc
            # This makes Q*bert "hover" near the disc waiting for Coily
            best_wait = None
            best_wait_dist = 99
            for a, nr, nc in neighbors(row, col):
                _, _, _, safe = is_move_safe(row, col, a, coily)
                if safe:
                    # Prefer moves that keep Q*bert close to the disc
                    d_to_disc = grid_distance(nr, nc, row, col)
                    if d_to_disc <= best_wait_dist:
                        best_wait_dist = d_to_disc
                        best_wait = a
            if best_wait is not None:
                return best_wait

        # Lure mode: navigate toward disc to kill Coily
        best_disc = min(discs_available,
            key=lambda d: grid_distance(row, col, d[0], d[1]))
        disc_dist = grid_distance(row, col, best_disc[0], best_disc[1])

        if not safe_with_followup and disc_dist <= 4 and coily_dist <= 3:
                if (row, col) != best_disc:
                    for blocked_set in [danger, set()]:
                        action = bfs_path_to(row, col, best_disc[0], best_disc[1], blocked_set)
                        if action is not None:
                            _, _, _, safe = is_move_safe(row, col, action, coily)
                            if safe:
                                return action

    # FLEE + ROUTE: unified scoring only when Coily is close
    # FLEE: Coily close — but NEVER flee to dead ends
    if coily and coily_dist <= 3:
        # Score all safe moves by combined value
        candidates = safe_with_followup if safe_with_followup else safe_moves
        if candidates:
            best_action = None
            best_score = -9999
            for entry in candidates:
                action, nr, nc, pc = entry[0], entry[1], entry[2], entry[3]
                esc = entry[4]
                d = grid_distance(nr, nc, pc[0], pc[1]) if pc else 99

                # Safety: distance from Coily + escape routes
                # Heavy penalty for moving adjacent to Coily (timing-unsafe on fast levels)
                coily_adjacent = coily_can_reach(coily, (nr, nc))
                safety = d * 2 + esc * 3 + (-20 if coily_adjacent else 0)

                # Productivity: strongly prefer uncolored cubes
                cube_val = 10 if not cube_done[nr][nc] else -3
                # Penalty for reverting done cubes on toggle levels
                if level >= 3 and cube_done[nr][nc]:
                    cube_val = -8
                # When no discs, bonus for undone cubes that are far from Coily
                # (lead Coily one way, color cubes the other way)
                if not discs_available and not cube_done[nr][nc]:
                    cube_coily_dist = grid_distance(nr, nc, coily[0], coily[1])
                    cube_val += cube_coily_dist * 2

                # Avoid dead ends — prefer positions with escape routes
                dead_end_penalty = -15 if esc == 0 else (-8 if esc == 1 else 0)
                if nr == MAX_ROW:
                    dead_end_penalty -= 12
                if (nr, nc) in ((5, 0), (5, 5)):
                    dead_end_penalty -= 25
                if (nc == 0 or nc == nr) and state.enemies:
                    dead_end_penalty -= 5

                # Disc proximity bonus when luring
                disc_bonus = 0
                if discs_available and coily_dist <= 4:
                    nearest_disc = min(discs_available,
                        key=lambda d_pos: grid_distance(nr, nc, d_pos[0], d_pos[1]))
                    if grid_distance(nr, nc, nearest_disc[0], nearest_disc[1]) < \
                       grid_distance(row, col, nearest_disc[0], nearest_disc[1]):
                        disc_bonus = 3

                score = safety + cube_val + dead_end_penalty + disc_bonus
                if score > best_score:
                    best_score = score
                    best_action = action
            if best_action is not None:
                return best_action

    # ROUTE: always use peel routing on L3+ (corners first, then inner cubes).
    # This ensures bottom-row/corner cubes are completed first, so Q*bert
    # can lead Coily around the center where there are more escape routes.
    # On second pass of non-toggle levels: no reversion penalty (cubes don't revert)
    peel_penalty = 0 if _on_second_pass else 4
    route_fn = (lambda r, c, cd, b=set(): bfs_peel_route(r, c, cd, b, peel_penalty)) if level >= 3 else bfs_nearest_undone
    route_fns = [route_fn] if route_fn == bfs_nearest_undone else [route_fn, bfs_nearest_undone]
    # Try routes with 3-step survival, then 2-step
    for survival_depth in [3, 2]:
        for fn in route_fns:
            for blocked_set in [danger, set()]:
                action = fn(row, col, cube_done, blocked_set)
                if action is not None:
                    nr, nc, pc, safe = is_move_safe(row, col, action, coily)
                    if safe and (not coily or survives_n_steps(nr, nc, pc, survival_depth)):
                        return action
    # All routes fail — prioritize safety (flee away from Coily)
    if coily and safe_moves:
        best = max(safe_moves, key=lambda x: (
            grid_distance(x[1], x[2], coily[0], coily[1]) * 3 + x[4]
            + (10 if not cube_done[x[1]][x[2]] else 0)
        ))
        return best[0]
    # Basic fallback: any safe route
    for fn in route_fns:
        action = fn(row, col, cube_done)
        if action is not None:
            _, _, _, safe = is_move_safe(row, col, action, coily)
            if safe:
                return action

    # Last resort: any safe move maximizing distance from Coily
    if safe_moves:
        if coily:
            best = max(safe_moves, key=lambda x: (
                grid_distance(x[1], x[2], coily[0], coily[1]) * 2 + x[4]
            ))
            return best[0]
        return safe_moves[0][0]

    # Emergency: navigate toward disc to lure Coily for a kill
    if coily_is_snake and discs_available:
        target_disc = min(discs_available, key=lambda d: grid_distance(row, col, d[0], d[1]))
        if (row, col) == target_disc and coily_dist <= 2:
            return DISCS[(row, col)]
        action = bfs_path_to(row, col, target_disc[0], target_disc[1])
        if action is not None:
            return action

    # Last-last resort: 1-step safety only (for levels where 2-step blocks everything)
    for fn in route_fns:
        action = fn(row, col, cube_done)
        if action is not None:
            dr, dc, _ = MOVES[action]
            nr, nc = row + dr, col + dc
            if not coily or (nr, nc) != coily:
                step1 = predict_coily_moves(coily[0], coily[1], nr, nc) if coily else []
                if (nr, nc) not in step1:
                    return action

    if valid_moves:
        if coily:
            best = max(valid_moves, key=lambda x: grid_distance(x[1], x[2], coily[0], coily[1]))
            return best[0]
        return valid_moves[0][0]
    return NOOP


def make_cube_grid():
    return [[False] * (r + 1) for r in range(MAX_ROW + 1)]


def run():
    global _no_progress_count, _on_second_pass, _prev_level_was_multi_hit
    def _get_arg(name, default):
        try:
            return int(sys.argv[sys.argv.index(name) + 1])
        except (ValueError, IndexError):
            return default
    speed = _get_arg("--speed", 1)
    start_level = _get_arg("--level", 1)
    fast = "--fast" in sys.argv
    if fast:
        speed = max(speed, 3)
    if speed > 1:
        import cv2
        env = gym.make("ALE/Qbert-v5", render_mode="rgb_array", repeat_action_probability=0.0)
        _render_counter = [0]
        _orig_step = env.step
        def _fast_step(action):
            result = _orig_step(action)
            _render_counter[0] += 1
            if _render_counter[0] % speed == 0:
                frame = env.render()
                cv2.imshow("Q*bert", frame[:, :, ::-1])
                cv2.waitKey(1)
            return result
        env.step = _fast_step
    else:
        env = gym.make("ALE/Qbert-v5", render_mode="human", repeat_action_probability=0.0)
    reader = QbertStateReader(env)
    episode = 0

    while True:
        episode += 1
        obs, info = env.reset()
        obs, _, done, info = reader.wait_for_game_start()
        if done:
            continue

        state = reader.read_state(obs, info)
        prev_lives = state.lives
        row, col = state.qbert if state.qbert else (0, 0)
        level = 1
        reader.set_level(level)
        cube_done = reader.read_cube_done()
        total_reward = 0
        jump_count = 0
        discs_available = set(DISCS.keys())
        prev_cubes_colored = reader.count_done_cubes()

        # Skip to target level by playing earlier levels without rendering
        if start_level > 1 and level < start_level:
            print(f"Skipping to level {start_level}...")
            # Temporarily disable rendering for speed
            orig_step = env.step
            if speed > 1:
                env.step = _orig_step  # bypass cv2 rendering
            else:
                # In human mode, swap to a non-rendering env for skip
                skip_env = gym.make("ALE/Qbert-v5", render_mode=None, repeat_action_probability=0.0)
                # Copy the game state
                state_ref = env.unwrapped.ale.cloneState()
                skip_env.reset()
                skip_env.unwrapped.ale.restoreState(state_ref)
                env.step = skip_env.step
                reader.env = skip_env
            while level < start_level and not done:
                cube_done = reader.read_cube_done()
                cubes_colored = reader.count_done_cubes()
                if cubes_colored >= NUM_CUBES:
                    reader.set_level(level)
                    cube_done = reader.read_cube_done()
                    cubes_colored = reader.count_done_cubes()
                action = pick_action(row, col, cube_done, state, discs_available, level)
                if action == NOOP:
                    for a, _, _ in neighbors(row, col):
                        action = a; break
                actual_pos = reader.read_qbert_position()
                if actual_pos: row, col = actual_pos
                using_disc = (row, col) in discs_available and action == DISCS.get((row, col))
                obs, r, t, tr, info = env.step(action)
                done = t or tr
                if not done:
                    obs, sr, done, info = reader.wait_for_landing(40 if using_disc else 15)
                    r += sr
                total_reward += r
                state = reader.read_state(obs, info)
                cube_done = reader.read_cube_done()
                cubes_colored = reader.count_done_cubes()
                if state.lives < prev_lives:
                    prev_lives = state.lives
                    if not done:
                        for _ in range(50):
                            obs, r2, t2, tr2, info = env.step(0)
                            total_reward += r2
                            if t2 or tr2: done = True; break
                        if not done:
                            obs, sr2, done, info = reader.wait_for_landing(40)
                            total_reward += sr2
                    state = reader.read_state(obs, info) if not done else state
                    row, col = state.qbert if state.qbert else (0, 0)
                    continue
                if using_disc: discs_available.discard((row, col))
                if state.qbert: row, col = state.qbert
                prev_lives = state.lives
                if cubes_colored >= NUM_CUBES:
                    level += 1
                    discs_available = set(DISCS.keys())
                    reader.set_level(level)
                    if not done:
                        obs, extra_r, done, info = reader.wait_for_level_start()
                        total_reward += extra_r
                    state = reader.read_state(obs, info) if not done else state
                    if not done and state.lives < prev_lives:
                        prev_lives = state.lives
                        for _ in range(50):
                            obs, r2, t2, tr2, info = env.step(0)
                            total_reward += r2
                            if t2 or tr2: done = True; break
                        if not done:
                            obs, sr2, done, info = reader.wait_for_landing(40)
                            total_reward += sr2
                        state = reader.read_state(obs, info) if not done else state
                    row, col = state.qbert if (not done and state.qbert) else (0, 0)
                    cube_done = reader.read_cube_done()
                    prev_cubes_colored = reader.count_done_cubes()
                    jump_count = 0
            env.step = orig_step  # restore rendering
            if speed <= 1:
                # Copy game state back to the rendering env
                state_ref = skip_env.unwrapped.ale.cloneState()
                env.unwrapped.ale.restoreState(state_ref)
                skip_env.close()
                reader.env = env
            print(f"Reached level {level}, lives: {state.lives}, score: {total_reward:.0f}")

        if episode == 1:
            print("Q*bert Agent (RAM + disc + peel routing)")
            print()
        print(f"--- Episode {episode}, Level {level} ---")

        while not done:
            cube_done = reader.read_cube_done()
            cubes_colored = reader.count_done_cubes()

            # Re-read actual position from RAM before every decision
            actual_pos = reader.read_qbert_position()
            if actual_pos:
                row, col = actual_pos  # Correct any drift

            action = pick_action(row, col, cube_done, state, discs_available, level)


            if action == NOOP:
                print(f"  !! STUCK: pos=({row},{col}) cubes={cubes_colored}/21 L{level} C={state.coily} E={state.enemies}")
                # Don't blindly default to DOWN — pick a safe valid move
                for alt_action, _, _ in neighbors(row, col):
                    action = alt_action
                    break
                else:
                    action = DOWN

            # Safety gate: verify the action leads to a valid position
            using_disc = (row, col) in discs_available and action == DISCS.get((row, col))
            if not using_disc and action in MOVES:
                dr, dc, _ = MOVES[action]
                nr, nc = row + dr, col + dc
                if not is_valid(nr, nc):
                    # Bad action — pick any valid neighbor instead
                    for alt_action, _, _ in neighbors(row, col):
                        action = alt_action
                        break
                # Extra safety: if moving DOWN from row 4+ and position wasn't
                # verified by RAM, don't risk it (could actually be at row 5)
                elif action in (DOWN, RIGHT) and nr >= MAX_ROW and not actual_pos:
                    for alt_action, anr, _ in neighbors(row, col):
                        if anr < row:  # prefer moving UP
                            action = alt_action
                            break
            move_name = MOVES.get(action, (0, 0, "disc"))[2] if action in MOVES else "disc"

            obs, r, t, tr, info = env.step(action)
            jump_reward = r
            done = t or tr
            if not done:
                obs, settle_r, done, info = reader.wait_for_landing(40 if using_disc else 15)
                jump_reward += settle_r

            total_reward += jump_reward
            jump_count += 1
            state = reader.read_state(obs, info, jump_reward, done)

            # Mark cube done by reward signal + learn target color
            if jump_reward >= 25 and not using_disc and state.qbert:
                reader.mark_cube_done_by_reward(state.qbert)
                reader.learn_target_color(state.qbert)

            # Re-read cubes from RAM after landing
            cube_done = reader.read_cube_done()
            cubes_colored = reader.count_done_cubes()

            # DEATH
            if state.lives < prev_lives:
                prev_lives = state.lives
                print(f"  DIED! Lives:{state.lives} Cubes:{cubes_colored}/{NUM_CUBES} Score:{total_reward:.0f}")
                if not done:
                    for _ in range(50):
                        obs, r, t, tr, info = env.step(NOOP)
                        total_reward += r
                        if t or tr: done = True; break
                    if not done:
                        obs, extra_r, done, info = reader.wait_for_landing(40)
                        total_reward += extra_r
                state = reader.read_state(obs, info) if not done else state
                row, col = state.qbert if state.qbert else (0, 0)
                prev_cubes_colored = cubes_colored
                continue

            # DISC USED
            if using_disc and state.lives >= prev_lives:
                discs_available.discard((row, col))
                label = "Coily killed!" if state.coily is None else "escape"
                print(f"  #{jump_count:3d} ** DISC ({row},{col})! {label} ** score:{total_reward:.0f}")
                row, col = state.qbert if state.qbert else (0, 0)
                prev_lives = state.lives
                prev_cubes_colored = cubes_colored
                continue

            # UPDATE POSITION (save previous for level transition detection)
            prev_row, prev_col = row, col
            if state.qbert:
                row, col = state.qbert
            else:
                dr, dc, _ = MOVES[action]
                nr, nc = row + dr, col + dc
                if is_valid(nr, nc):
                    row, col = nr, nc

            # LOG: always log every move for debugging
            cs = f" C={state.coily}" if state.coily else ""
            ds = f" discs={len(discs_available)}" if discs_available else ""
            tgt = f" tgt={reader._cube_target_color}" if reader._cube_target_color else " tgt=?"
            if cubes_colored > prev_cubes_colored:
                pl = f" peel={PEEL_LAYERS[(row,col)]}" if level >= 3 else ""
                print(f"  #{jump_count:3d} {move_name:>11s}->({row},{col}) cubes:{cubes_colored:2d}/{NUM_CUBES} score:{total_reward:.0f}{cs}{ds}{pl}{tgt}")
            elif jump_count % 5 == 0 or cubes_colored < prev_cubes_colored:
                print(f"  #{jump_count:3d} {move_name:>11s}->({row},{col}) cubes:{cubes_colored}/{NUM_CUBES} score:{total_reward:.0f} r={jump_reward:.0f}{cs}{tgt}")
            elif cubes_colored < prev_cubes_colored and level >= 3:
                print(f"  #{jump_count:3d} REVERTED ({row},{col}) cubes:{cubes_colored}/{NUM_CUBES}")
            prev_cubes_colored = cubes_colored

            # LEVEL COMPLETE: check RAM[0] (hw $80) for value 1.
            # This ONLY appears during the level completion bonus cycle.
            # Much more reliable than cubes_colored (celebration flash corrupts it).
            game_state = env.unwrapped.ale.getRAM()[0]
            game_level_complete = (game_state == 1)

            if game_level_complete:
                # Wait for the bonus to finish
                for _ in range(50):
                    obs_c, r_c, t_c, tr_c, info_c = env.step(NOOP)
                    total_reward += r_c
                    if t_c or tr_c: done = True; break
                if not done:
                    state = reader.read_state(obs_c, info_c)

            # No multi-hit detection needed — RAM[0]==1 only fires when
            # the game truly considers the level complete (all passes done).

            if game_level_complete:
                print(f"\n  === LEVEL {level} COMPLETE! Score: {total_reward:.0f} ===\n")
                level += 1
                _on_second_pass = False
                jump_count = 0
                prev_lives = state.lives
                discs_available = set(DISCS.keys())
                if not done:
                    # Spam first move until level starts — baseline captured inside
                    obs, extra_r, done, info = reader.wait_for_level_start()
                    total_reward += extra_r
                reader.set_level(level)  # snapshot baseline AFTER transition
                # After multi-hit levels: use majority cube value as target
                state = reader.read_state(obs, info) if not done else state
                # Handle death during level transition
                if not done and state.lives < prev_lives:
                    prev_lives = state.lives
                    for _ in range(50):
                        obs, r, t, tr, info = env.step(NOOP)
                        total_reward += r
                        if t or tr: done = True; break
                    if not done:
                        obs, extra_r, done, info = reader.wait_for_landing(40)
                        total_reward += extra_r
                    state = reader.read_state(obs, info) if not done else state
                row, col = state.qbert if (not done and state.qbert) else (0, 0)
                cube_done = reader.read_cube_done()
                prev_cubes_colored = reader.count_done_cubes()
                print(f"--- Level {level} ---")
                continue

            prev_lives = state.lives

        print(f"Game over! Score: {total_reward:.0f}, Level {level}\n")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
