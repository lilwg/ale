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
    # Unsafe if ANY of Coily's possible chase moves lands on Q*bert's destination
    possible_moves = predict_coily_moves(coily[0], coily[1], nr, nc)
    if (nr, nc) in possible_moves:
        return nr, nc, coily, False
    # Return the most likely predicted position for downstream use
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

    # Danger zone: all enemies from RAM + their neighbors + predicted next positions
    # Enemies bounce DOWN the pyramid, so also block one row below each enemy
    danger = set()
    for epos in state.enemies:
        danger.add(epos)
        for _, nr, nc in neighbors(epos[0], epos[1]):
            danger.add((nr, nc))
        # Predict next bounce: enemies move DOWN (row+1), so block 2 rows ahead
        for _, nr, nc in neighbors(epos[0], epos[1]):
            if nr > epos[0]:  # downward neighbor
                danger.add((nr, nc))
                for _, nnr, nnc in neighbors(nr, nc):
                    if nnr > nr:  # 2 rows down
                        danger.add((nnr, nnc))
    # Ensure Coily (pixel-detected, most reliable) is always in danger
    if coily:
        danger.add(coily)
        for _, nr, nc in neighbors(coily[0], coily[1]):
            danger.add((nr, nc))

    coily_dist = grid_distance(row, col, coily[0], coily[1]) if coily else 99

    # Pre-compute safe moves with 2-hop lookahead + escape route scoring
    valid_moves = []
    for action, nr, nc in neighbors(row, col):
        _, _, predicted_coily, safe = is_move_safe(row, col, action, coily)
        followup = has_safe_followup(row, col, action, coily) if safe else False
        escapes = count_escape_routes(nr, nc, predicted_coily if safe else coily)
        valid_moves.append((action, nr, nc, safe, predicted_coily, followup, escapes))
    safe_moves = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, _, esc in valid_moves if s]
    safe_with_followup = [(a, nr, nc, pc, esc) for a, nr, nc, s, pc, f, esc in valid_moves if s and f]

    # Count remaining cubes
    cubes_remaining = sum(
        1 for r in range(MAX_ROW + 1) for c in range(r + 1) if not cube_done[r][c]
    )

    # FINISH LEVEL: if few cubes left, rush to complete (but never onto Coily)
    if cubes_remaining <= 3:
        action = bfs_nearest_undone(row, col, cube_done)
        if action is not None:
            _, _, _, safe = is_move_safe(row, col, action, coily)
            if safe:
                return action

    # --- DISC LURE STRATEGY ---
    # When Coily is active and a disc is available, try to lure Coily to disc for a kill.
    # Fire disc only when Coily is close enough to follow us off the edge.
    if coily and discs_available:
        on_disc = (row, col) in discs_available

        # Fire disc: Coily adjacent (guaranteed kill) or truly cornered
        if on_disc and (coily_dist <= 1 or not safe_with_followup):
            return DISCS[(row, col)]

        # Lure mode: navigate toward disc when Coily is active
        # Pick the disc that's closest to BOTH us and Coily (best lure path)
        if coily_dist <= 5:
            best_disc = min(discs_available,
                key=lambda d: grid_distance(row, col, d[0], d[1])
                            + grid_distance(coily[0], coily[1], d[0], d[1]) * 0.3)
            disc_dist = grid_distance(row, col, best_disc[0], best_disc[1])

            # If we're cornered (no safe followup) or Coily is very close, prioritize disc
            if not safe_with_followup or (coily_dist <= 2 and disc_dist <= 2):
                if (row, col) != best_disc:
                    for blocked_set in [danger, set()]:
                        action = bfs_path_to(row, col, best_disc[0], best_disc[1], blocked_set)
                        if action is not None:
                            _, _, _, safe = is_move_safe(row, col, action, coily)
                            if safe:
                                return action

    # FLEE + ROUTE: unified scoring when Coily is present
    if coily and coily_dist <= 5:
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

                # Productivity: prefer uncolored cubes
                cube_val = 5 if not cube_done[nr][nc] else 0
                # Penalty for reverting done cubes on toggle levels
                if level >= 3 and cube_done[nr][nc]:
                    cube_val = -8

                # Avoid dead ends: heavy penalty for corners/bottom with few escapes
                dead_end_penalty = -15 if esc == 0 else (-8 if esc == 1 else 0)

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

    # ROUTE: peel-based Dijkstra for L3+ (outside-in, reversion penalty),
    # simple BFS for L1-2
    route_fn = bfs_peel_route if level >= 3 else bfs_nearest_undone
    for blocked_set in [danger, set()]:
        action = route_fn(row, col, cube_done, blocked_set)
        if action is not None:
            nr, nc, _, safe = is_move_safe(row, col, action, coily)
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

    # Emergency: use disc if available
    if coily and discs_available:
        target_disc = min(discs_available, key=lambda d: grid_distance(row, col, d[0], d[1]))
        if (row, col) == target_disc:
            return DISCS[(row, col)]
        action = bfs_path_to(row, col, target_disc[0], target_disc[1])
        if action is not None:
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

        if episode == 1:
            print("Q*bert Agent (RAM + disc + peel routing)")
            print()
        print(f"--- Episode {episode}, Level {level} ---")

        while not done:
            cube_done = reader.read_cube_done()
            cubes_colored = reader.count_done_cubes()

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

            # UPDATE POSITION
            if state.qbert:
                row, col = state.qbert
            else:
                dr, dc, _ = MOVES[action]
                nr, nc = row + dr, col + dc
                if is_valid(nr, nc):
                    row, col = nr, nc

            # LOG CUBE PROGRESS
            if cubes_colored > prev_cubes_colored:
                cs = ""
                if state.coily:
                    d = grid_distance(row, col, state.coily[0], state.coily[1])
                    cs = f" C={state.coily} d={d}"
                ds = f" discs={len(discs_available)}" if discs_available else ""
                pl = f" peel={PEEL_LAYERS[(row,col)]}" if level >= 3 else ""
                print(f"  #{jump_count:3d} {move_name:>11s}->({row},{col}) cubes:{cubes_colored:2d}/{NUM_CUBES} score:{total_reward:.0f}{cs}{ds}{pl}")
            elif cubes_colored < prev_cubes_colored and level >= 3:
                print(f"  #{jump_count:3d} REVERTED ({row},{col}) cubes:{cubes_colored}/{NUM_CUBES}")
            prev_cubes_colored = cubes_colored

            # LEVEL COMPLETE
            if cubes_colored >= NUM_CUBES:
                print(f"\n  === LEVEL {level} COMPLETE! Score: {total_reward:.0f} ===\n")
                level += 1
                jump_count = 0
                prev_lives = state.lives
                discs_available = set(DISCS.keys())
                reader.set_level(level)  # sets color cycle
                if not done:
                    # Spam first move until level starts — baseline captured inside
                    obs, extra_r, done, info = reader.wait_for_level_start()
                    total_reward += extra_r
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
