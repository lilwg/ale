"""Run heuristic agent headlessly and log detailed state for debugging."""
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

from qbert_state import QbertStateReader, is_valid, MAX_ROW, NUM_CUBES, CUBE_RAM
from qbert_heuristic import pick_action, MOVES, DISCS, NOOP, DOWN, PEEL_LAYERS, grid_distance

env = gym.make("ALE/Qbert-v5", render_mode=None, repeat_action_probability=0.0)
reader = QbertStateReader(env)

MAX_EPISODES = 3

for episode in range(1, MAX_EPISODES + 1):
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

    print(f"\n=== Episode {episode}, Level {level} ===")
    print(f"  Initial color: {reader._cube_initial_color}")
    print(f"  Start pos: ({row},{col}), lives: {state.lives}")

    while not done:
        cube_done = reader.read_cube_done()
        cubes_colored = reader.count_done_cubes()

        action = pick_action(row, col, cube_done, state, discs_available, level)
        orig_action = action
        if action == NOOP:
            action = DOWN

        using_disc = (row, col) in discs_available and action == DISCS.get((row, col))
        move_name = MOVES.get(action, (0, 0, "noop"))[2] if action in MOVES else "disc"

        # Log every move
        coily_info = f" C={state.coily}" if state.coily else ""
        green_info = f" G={state.green}" if state.green else ""
        if orig_action == NOOP:
            print(f"  #{jump_count+1:3d} NOOP->DOWN from ({row},{col}){coily_info}")

        obs, r, t, tr, info = env.step(action)
        jump_reward = r
        done = t or tr
        if not done:
            obs, settle_r, done, info = reader.wait_for_landing(40 if using_disc else 15)
            jump_reward += settle_r

        total_reward += jump_reward
        jump_count += 1
        state = reader.read_state(obs, info, jump_reward, done)

        cube_done = reader.read_cube_done()
        cubes_colored = reader.count_done_cubes()

        # DEATH
        if state.lives < prev_lives:
            prev_lives = state.lives
            coily_at = f" Coily was at {state.coily}" if state.coily else ""
            print(f"  #{jump_count:3d} *** DIED *** at ({row},{col}) {move_name} lives:{state.lives} cubes:{cubes_colored}/{NUM_CUBES} score:{total_reward:.0f}{coily_at}")
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
        old_row, old_col = row, col
        if state.qbert:
            row, col = state.qbert
        else:
            dr, dc, _ = MOVES[action]
            nr, nc = row + dr, col + dc
            if is_valid(nr, nc):
                row, col = nr, nc
            print(f"  #{jump_count:3d} WARNING: RAM position read failed, dead reckoning to ({row},{col})")

        # LOG
        if cubes_colored > prev_cubes_colored:
            cs = f" C={state.coily}" if state.coily else ""
            print(f"  #{jump_count:3d} {move_name:>11s} ({old_row},{old_col})->({row},{col}) cubes:{cubes_colored:2d}/{NUM_CUBES} score:{total_reward:.0f}{cs}")
        elif cubes_colored < prev_cubes_colored and level >= 3:
            print(f"  #{jump_count:3d} REVERTED ({row},{col}) cubes:{cubes_colored}/{NUM_CUBES}")
        prev_cubes_colored = cubes_colored

        # LEVEL COMPLETE
        if cubes_colored >= NUM_CUBES:
            print(f"\n  === LEVEL {level} COMPLETE! jumps:{jump_count} Score: {total_reward:.0f} ===\n")
            level += 1
            jump_count = 0
            prev_lives = state.lives
            discs_available = set(DISCS.keys())
            reader.set_level(level)
            if not done:
                obs, extra_r, done, info = reader.wait_for_level_start()
                total_reward += extra_r
            state = reader.read_state(obs, info) if not done else state
            if not done and state.lives < prev_lives:
                prev_lives = state.lives
                print(f"  ** DIED during level transition **")
                for _ in range(50):
                    obs, r2, t2, tr2, info = env.step(0)
                    total_reward += r2
                    if t2 or tr2: done = True; break
                if not done:
                    obs, extra_r, done, info = reader.wait_for_landing(40)
                    total_reward += extra_r
                state = reader.read_state(obs, info) if not done else state
            row, col = state.qbert if (not done and state.qbert) else (0, 0)
            cube_done = reader.read_cube_done()
            prev_cubes_colored = reader.count_done_cubes()
            state = reader.read_state(obs, info) if not done else state
            print(f"--- Level {level}, initial_color: {reader._cube_initial_color} ---")
            continue

        prev_lives = state.lives

    print(f"\nGame over! Score: {total_reward:.0f}, Level {level}")

env.close()
