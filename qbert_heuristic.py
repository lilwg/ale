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
import gymnasium as gym
import ale_py
import numpy as np
from collections import deque

from qbert_state import (
    QbertStateReader, is_valid, MAX_ROW, NUM_CUBES,
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
    best = (cr, cc)
    best_dist = grid_distance(cr, cc, qr, qc)
    for _, nr, nc in neighbors(cr, cc):
        d = grid_distance(nr, nc, qr, qc)
        if d < best_dist:
            best_dist = d
            best = (nr, nc)
    return best


def is_move_safe(row, col, action, coily):
    dr, dc, _ = MOVES[action]
    nr, nc = row + dr, col + dc
    if not is_valid(nr, nc):
        return nr, nc, coily, False
    if coily is None:
        return nr, nc, None, True
    pcr, pcc = predict_coily_move(coily[0], coily[1], nr, nc)
    safe = (pcr, pcc) != (nr, nc)
    return nr, nc, (pcr, pcc), safe


def has_safe_followup(row, col, action, coily):
    """After taking action, does at least one safe 2nd move exist?"""
    nr, nc, predicted_coily, safe = is_move_safe(row, col, action, coily)
    if not safe or not is_valid(nr, nc):
        return False
    for _, nnr, nnc in neighbors(nr, nc):
        if predicted_coily is None:
            return True
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
        if (nr, nc) not in seen and (nr, nc) not in blocked:
            if not cube_done[nr][nc]:
                return action
            seen.add((nr, nc))
            queue.append((nr, nc, action))
    while queue:
        cr, cc, first_action = queue.popleft()
        for _, nr, nc in neighbors(cr, cc):
            if (nr, nc) not in seen and (nr, nc) not in blocked:
                if not cube_done[nr][nc]:
                    return first_action
                seen.add((nr, nc))
                queue.append((nr, nc, first_action))
    return None


def bfs_peel_route(row, col, cube_done, blocked=set()):
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
            if (nr, nc) in blocked:
                continue
            step_cost = 1
            if cube_done[nr][nc]:
                step_cost += 4
            new_cost = cost + step_cost
            if new_cost < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = new_cost
                fa = first_action if first_action is not None else action
                heapq.heappush(heap, (new_cost, nr, nc, fa))
    return None


def pick_action(row, col, cube_done, state, discs_available, level=1):
    coily = state.coily
    green = state.green

    # Danger zone: enemies + their neighbors
    danger = set()
    if coily:
        danger.add(coily)
        for _, nr, nc in neighbors(coily[0], coily[1]):
            danger.add((nr, nc))
    if green:
        danger.add(green)
        for _, nr, nc in neighbors(green[0], green[1]):
            danger.add((nr, nc))

    coily_dist = grid_distance(row, col, coily[0], coily[1]) if coily else 99

    # Pre-compute safe moves with 2-hop lookahead
    valid_moves = []
    for action, nr, nc in neighbors(row, col):
        _, _, predicted_coily, safe = is_move_safe(row, col, action, coily)
        followup = has_safe_followup(row, col, action, coily) if safe else False
        valid_moves.append((action, nr, nc, safe, predicted_coily, followup))
    safe_moves = [(a, nr, nc, pc) for a, nr, nc, s, pc, _ in valid_moves if s]
    safe_with_followup = [(a, nr, nc, pc) for a, nr, nc, s, pc, f in valid_moves if s and f]

    # DISC: fire if on disc square and Coily close
    if coily and discs_available and coily_dist <= 2:
        if (row, col) in discs_available:
            return DISCS[(row, col)]

    # DISC: navigate toward disc when Coily close
    if coily and discs_available and coily_dist <= 2:
        target_disc = min(discs_available, key=lambda d: grid_distance(row, col, d[0], d[1]))
        if (row, col) != target_disc:
            for blocked_set in [danger, set()]:
                action = bfs_path_to(row, col, target_disc[0], target_disc[1], blocked_set)
                if action is not None:
                    _, _, _, safe = is_move_safe(row, col, action, coily)
                    if safe:
                        return action

    # FLEE: Coily close — prefer moves with safe followup (anti-cornering)
    if coily and coily_dist <= 3:
        flee_moves = safe_with_followup if safe_with_followup else safe_moves
        if flee_moves:
            best_action = None
            best_score = -999
            for action, nr, nc, pc in flee_moves:
                d = grid_distance(nr, nc, pc[0], pc[1]) if pc else 99
                bonus = 3 if not cube_done[nr][nc] else 0
                penalty = -5 if (nr, nc) in danger else 0
                score = d + bonus + penalty
                if score > best_score:
                    best_score = score
                    best_action = action
            if best_action is not None:
                return best_action

    # ROUTE: peel-based Dijkstra for L3+ (outside-in, reversion penalty),
    # simple BFS for L1-2
    route_fn = bfs_peel_route if level >= 3 else bfs_nearest_undone
    for blocked_set in [danger, set()]:
        action = route_fn(row, col, cube_done, blocked_set)
        if action is not None:
            _, _, _, safe = is_move_safe(row, col, action, coily)
            if safe:
                return action

    # Fallback: route without safety check
    action = route_fn(row, col, cube_done)
    if action is not None:
        return action

    # All done or stuck: flee
    if safe_moves:
        if coily:
            best = max(safe_moves, key=lambda x: grid_distance(x[1], x[2], coily[0], coily[1]))
            return best[0]
        return safe_moves[0][0]
    if valid_moves:
        if coily:
            best = max(valid_moves, key=lambda x: grid_distance(x[1], x[2], coily[0], coily[1]))
            return best[0]
        return valid_moves[0][0]
    return NOOP


def make_cube_grid():
    return [[False] * (r + 1) for r in range(MAX_ROW + 1)]


def run():
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
        cube_done = make_cube_grid()
        cube_done[row][col] = True
        total_reward = 0
        cubes_colored = 0
        jump_count = 0
        level = 1
        discs_available = set(DISCS.keys())

        if episode == 1:
            print("Q*bert Agent (RAM + disc + peel routing)")
            print()
        print(f"--- Episode {episode}, Level {level} ---")

        while not done:
            action = pick_action(row, col, cube_done, state, discs_available, level)
            if action == NOOP:
                action = DOWN

            using_disc = (row, col) in discs_available and action == DISCS.get((row, col))
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
                continue

            # DISC USED
            if using_disc and state.lives >= prev_lives:
                discs_available.discard((row, col))
                label = "Coily killed!" if state.coily is None else "escape"
                print(f"  #{jump_count:3d} ** DISC ({row},{col})! {label} ** score:{total_reward:.0f}")
                row, col = state.qbert if state.qbert else (0, 0)
                prev_lives = state.lives
                continue

            # UPDATE POSITION
            if state.qbert:
                row, col = state.qbert
            else:
                dr, dc, _ = MOVES[action]
                nr, nc = row + dr, col + dc
                if is_valid(nr, nc):
                    row, col = nr, nc

            # UPDATE CUBE STATE
            was_undone = not cube_done[row][col]
            if jump_reward >= 25:
                cube_done[row][col] = True
                cubes_colored += 1
            elif level >= 3 and cube_done[row][col]:
                # Toggle level: stepping on done cube without reward = reverted
                cube_done[row][col] = False
                cubes_colored = max(0, cubes_colored - 1)
                print(f"  #{jump_count:3d} REVERTED ({row},{col}) cubes:{cubes_colored}/{NUM_CUBES}")

            if was_undone and cube_done[row][col]:
                cs = ""
                if state.coily:
                    d = grid_distance(row, col, state.coily[0], state.coily[1])
                    cs = f" C={state.coily} d={d}"
                ds = f" discs={len(discs_available)}" if discs_available else ""
                pl = f" peel={PEEL_LAYERS[(row,col)]}" if level >= 3 else ""
                print(f"  #{jump_count:3d} {move_name:>11s}->({row},{col}) cubes:{cubes_colored:2d}/{NUM_CUBES} score:{total_reward:.0f}{cs}{ds}{pl}")

            # LEVEL COMPLETE
            if cubes_colored >= NUM_CUBES:
                print(f"\n  === LEVEL {level} COMPLETE! Score: {total_reward:.0f} ===\n")
                level += 1
                cubes_colored = 0
                cube_done = make_cube_grid()
                jump_count = 0
                prev_lives = state.lives
                discs_available = set(DISCS.keys())
                if not done:
                    # Wait for level transition animation
                    for _ in range(80):
                        obs, r, t, tr, info = env.step(NOOP)
                        total_reward += r
                        if t or tr: done = True; break
                    if not done:
                        obs, extra_r, done, info = reader.wait_for_landing(60)
                        total_reward += extra_r
                # Force (0,0) — Q*bert always starts at top after level transition
                row, col = 0, 0
                cube_done[0][0] = True
                state = reader.read_state(obs, info) if not done else state
                print(f"--- Level {level} ---")
                continue

            prev_lives = state.lives

        print(f"Game over! Score: {total_reward:.0f}, Level {level}\n")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
