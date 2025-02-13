import os
import random
from collections import deque

import numpy as np
import pandas as pd
import retro
import torch
from agents.base import BaseAgent
from agents.dqn_recurrent_agent import RecurrentDQNAgent
from inputs import bit_to_input, input_to_bit
from preprocess import preprocess_image, stack_frames
from reward import compute_reward
from rich.console import Console
from utils import pretty_print_info as pprint

console = Console()

# =============================================================================
# 1. ENVIRONMENT FACTORY COMPONENTS
# =============================================================================


def create_training_env(room: str, render_mode: str, record: str):
    """
    Creates and resets a Retro environment for live training.
    """
    if room is None:
        raise ValueError(
            "The game/room identifier (room) must be provided and cannot be None."
        )
    if record is None:
        raise ValueError("The record folder path must be provided and cannot be None.")
    env = retro.make(game=room, render_mode=render_mode, record=record)
    obs, info = env.reset()
    return env, obs, info


def create_replay_env(bk2_file: str, render_mode: str, record: str):
    """
    Loads a Retro movie (BK2 file) and creates an environment in replay mode.
    """
    if bk2_file is None:
        raise ValueError(
            "bk2_file cannot be None. Please supply a valid BK2 file path."
        )

    movie = retro.Movie(bk2_file)

    movie.step()  # initialize movie state
    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        render_mode=render_mode,
        record=record,
    )
    env.initial_state = movie.get_state()
    env.statename = "demo_state"
    env.reset()

    return movie, env


# =============================================================================
# 2. RUN LOOP COMPONENTS
# =============================================================================


def store_replay_episode(
    movie, env, agent, frame_stack_size: int, frame_skip: int, replay: bool = False
):
    """
    Runs one replay (demonstration) episode using the given Retro movie.

    This version mimics the live mode loop structure:
      - Instead of selecting an action via agent.act(), we read the demonstration action from the movie.

    Returns:
        episode_reward: The total reward accumulated in the episode.
        step_count: The total number of steps processed.
    """
    # Reset environment and initialize frame stack.
    obs, info = env.reset()
    processed_obs = preprocess_image(obs)
    frame_buffer = deque(
        [processed_obs for _ in range(frame_stack_size)], maxlen=frame_stack_size
    )
    state = stack_frames(frame_buffer)

    # Initialize tracking variables.
    score = 0
    round_reward = 0
    episode_reward = 0
    step_count = 0
    round_count = 0
    neutral_action = input_to_bit(0)
    info = initial_info = {
        "health": 176,
        "enemy_health": 176,
        "matches_won": 0,
        "enemy_matches_won": 0,
    }

    done = False
    truncated = False

    while movie.step():
        keys = []

        # EXTRACT ACTION FROM REPLAY STEP
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))

        keys_int = [int(i) for i in keys]
        demo_action = bit_to_input(keys_int)

        # PROCESS ENV STEP
        next_obs, next_score, done, truncated, next_info = env.step(keys)
        step_count += 1

        # COMPUTE REWARDS & STORE
        processed_next = preprocess_image(next_obs)
        frame_buffer.append(processed_next)
        next_state = stack_frames(frame_buffer)

        # Compute reward using the same function as live mode.
        reward = compute_reward(info, next_info, score, next_score, step_count)

        # Store the transition and update the agent.
        agent.remember(
            state.numpy(), demo_action, reward, next_state.numpy(), done or truncated
        )

        if replay:
            # execute training step
            agent.replay()

        # Update our tracking variables.
        state = next_state
        score = next_score
        info = next_info
        episode_reward += reward
        round_reward += reward

        round_timer = step_count / 60.0  # assuming 60 fps

        # SKIP STORING FRAMES OUTSIDES MATCHES
        # Check for round end conditions.
        if info["health"] < 0 or info["enemy_health"] < 0 or round_timer >= 100:
            agent.reset()
            skip_info = info
            while skip_info["health"] < 0 or skip_info["enemy_health"] < 0:
                _, _, _, _, skip_info = env.step(neutral_action)

            round_count += 1
            round_reward = 0
            step_count = 0
            info = initial_info
            state = stack_frames(frame_buffer)

        if info.get("enemy_matches_won", 0) == 2 or done or truncated:
            break  # Exit the frame skip loop if the round ends.

    return episode_reward, step_count


def run_replay_mode(
    agent,
    bk2_folder: str,
    frame_stack_size: int,
    frame_skip: int,
    render_mode: str,
    record: str = None,
):
    """
    Iterates over all BK2 files in the given folder.
    For each movie, the agent learns from the demonstration.
    """
    if bk2_folder is None:
        raise ValueError(
            "bk2_folder is None. Please provide a valid folder path containing .bk2 files."
        )

    bk2_files = [
        os.path.join(bk2_folder, f)
        for f in os.listdir(bk2_folder)
        if f.endswith(".bk2")
    ]
    if not bk2_files:
        raise ValueError(f"No .bk2 files found in folder: {bk2_folder}")

    replay_results = []
    for bk2_file in bk2_files:
        console.print(f"[bold yellow]Replaying movie: {bk2_file}[/bold yellow]")
        movie, env = create_replay_env(bk2_file, render_mode, record)
        episode_reward, steps = store_replay_episode(
            movie, env, agent, frame_stack_size, frame_skip
        )
        replay_results.append(
            {"bk2_file": bk2_file, "reward": episode_reward, "steps": steps}
        )
        env.close()

    return replay_results


def run_live_mode(
    agent,
    room: str,
    render_mode: str,
    record: str,
    episodes: int,
    frame_stack_size: int,
    frame_skip: int,
    checkpoint_freq: int,
    act_kwargs: dict = {},
    act_validate_kwargs: dict = {},
    act_validate_step_freq: int = 10,
    warm_start: int = 10,
):
    """
    Runs the live (agent-controlled) loop.
    The agent selects actions via its policy.
    """
    env, initial_frame, info = create_training_env(room, render_mode, record)
    processed_initial = preprocess_image(initial_frame)
    frame_buffer = deque(
        [processed_initial for _ in range(frame_stack_size)], maxlen=frame_stack_size
    )
    state = stack_frames(frame_buffer)

    neutral_action = input_to_bit(0)
    episode_results = []
    score = 0
    initial_info = {
        "health": 176,
        "enemy_health": 176,
        "matches_won": 0,
        "enemy_matches_won": 0,
    }
    episode_count = 0

    while episode_count < episodes:
        console.print(
            f"[bold magenta]--- BEGIN EPISODE {episode_count} ---[/bold magenta]"
        )
        done = False
        truncated = False
        episode_reward = 0
        round_reward = 0
        step_count = 0
        round_count = 0
        info = initial_info

        kwargs = act_kwargs
        if episode_count % act_validate_step_freq == 0:
            kwargs = act_validate_kwargs

        while not done and not truncated:
            agent_action = agent.act(state.numpy(), **kwargs)
            for _ in range(frame_skip):
                action = input_to_bit(agent_action)
                next_obs, next_score, done, truncated, next_info = env.step(action)
                step_count += 1

                processed_next = preprocess_image(next_obs)
                frame_buffer.append(processed_next)
                next_state = stack_frames(frame_buffer)

                reward = compute_reward(info, next_info, score, next_score, step_count)

                if episode_count > warm_start:
                    agent.remember(
                        state.numpy(),
                        agent_action,
                        reward,
                        next_state.numpy(),
                        done or truncated,
                    )

                agent.replay()

                state = next_state
                score = next_score
                info = next_info
                episode_reward += reward
                round_reward += reward

                round_timer = step_count / 60.0
                if info["health"] < 0 or info["enemy_health"] < 0 or round_timer >= 100:
                    agent.reset()
                    skip_info = info
                    while skip_info["health"] < 0 or skip_info["enemy_health"] < 0:
                        _, _, _, _, skip_info = env.step(neutral_action)
                    episode_result = {
                        "episode": episode_count,
                        "reward": episode_reward,
                        "round_reward": round_reward,
                        "round": round_count,
                        "win?": info["health"] > info["enemy_health"],
                        "agent_epsilon": agent.epsilon,
                        "timer": round_timer,
                        "step_count": step_count,
                    }
                    episode_results.append(episode_result)
                    pprint(episode_result)
                    step_count = 0
                    round_reward = 0
                    round_count += 1
                    state = stack_frames(frame_buffer)
                if info.get("enemy_matches_won", 0) == 2 or done or truncated:
                    break
        episode_count += 1
        if episode_count % checkpoint_freq == 0:
            agent.save("./tmp/dqn_checkpoint.pth")
            pd.DataFrame(episode_results).to_csv("./tmp/tmp_results.csv")

        initial_frame, info = env.reset()
        processed_initial = preprocess_image(initial_frame)
        frame_buffer = deque(
            [processed_initial for _ in range(frame_stack_size)],
            maxlen=frame_stack_size,
        )
        state = stack_frames(frame_buffer)

    console.print("[bold red]--- LIVE PLAY FINISHED ---[/bold red]")
    agent.save()
    env.close()
    return episode_results


# =============================================================================
# 3. MAIN ORCHESTRATION
# =============================================================================


def play_game(
    episodes: int = 100,
    checkpoint_freq: int = 50,
    frame_stack_size: int = 4,
    frame_skip: int = 1,
    agent: BaseAgent = None,
    room: str = "StreetFighterIISpecialChampionEdition-Genesis",
    render_mode: str = "rgb_array",
    record: str = "./",
    bk2_folder: str = "/Users/everton.soutolima/Repository/street-fighter-ai/replays/human/",
    seed: int = 236,  # hadouken!
):
    """
    Orchestrates the learning process.

    1. If a bk2_folder is provided (and not None), the agent first runs in replay mode—learning
       from demonstration (each BK2 file is replayed).
    2. After replay mode is exhausted, the loop switches to live play mode,
       where the agent selects actions using its policy.

    Returns a dictionary with both replay and live play results.
    """

    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Instantiate agent if not provided.
    if agent is None:
        action_dim = 576
        agent = RecurrentDQNAgent(
            action_dim=action_dim, state_shape=(frame_stack_size, 96, 96)
        )

    results = {}
    if bk2_folder is not None:
        # fills the agent's memory buffer with pre-recorded expert gameplay
        console.print("[bold green]--- STARTING REPLAY MODE ---[/bold green]")

        run_replay_mode(
            agent, bk2_folder, frame_stack_size, frame_skip, render_mode, record
        )
        console.print("[bold green]--- REPLAY MODE FINISHED ---[/bold green]")

    console.print("[bold green]--- STARTING LIVE PLAY MODE ---[/bold green]")
    live_results = run_live_mode(
        agent,
        room,
        render_mode,
        record,
        episodes,
        frame_stack_size,
        frame_skip,
        checkpoint_freq,
        act_kwargs={"top_k": agent.action_dim, "temperature": 1.0},
        act_validate_kwargs={"top_k": 1},
    )
    results["live"] = live_results

    return results
