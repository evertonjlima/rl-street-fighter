from collections import deque

import numpy as np
import retro
from agents.base import BaseAgent
from agents.dqn_agent import DQNAgent
from preprocess import rgb2gray_luminance, stack_frames_grayscale
from rich.console import Console
from utils import pretty_print_info as pprint

console = Console()


def game_input_map(x: int):
    """
    Returns a 12-bit array representing a "simplified" set of
    moves for Ryu. We include:
      1) Basic directions: NEUTRAL, LEFT, RIGHT, DOWN, UP
      2) Basic attacks: LIGHT_PUNCH, MEDIUM_PUNCH, HEAVY_PUNCH,
                       LIGHT_KICK,  MEDIUM_KICK,  HEAVY_KICK
      3) A few direction+attack combos (crouching / jumping attacks)

    The index 'x' can range from 0 up to (len(button_map)-1).

    The 12 bits correspond to Street Fighter button order from
    your original code snippet:
        Index:  0   1   2   3   4   5   6   7   8   9   10  11
        Label: MK  LK  ?   ?   UP  DN  RT  LT  HK  MP  LP  HP
        (The ? bits are unused in your snippet.)

    Adjust if your environment differs!
    """

    # Helper to combine (OR) two 12-bit lists
    def or_bits(a, b):
        return [int(i or j) for i, j in zip(a, b)]

    # -------------------------------------------------------
    # 1) Define Single Directions in 12-bit format
    #    (matching your original snippet’s indexing)
    # -------------------------------------------------------
    NEUTRAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    LEFT = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    RIGHT = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    DOWN = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    UP = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # -------------------------------------------------------
    # 2) Define Single Attack Buttons
    # -------------------------------------------------------
    LIGHT_PUNCH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    MEDIUM_PUNCH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    HEAVY_PUNCH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    LIGHT_KICK = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    MEDIUM_KICK = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    HEAVY_KICK = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    # -------------------------------------------------------
    # 3) Create "direction + attack" combos you want to allow.
    #    (Below: just a few examples for crouching and jumping.)
    # -------------------------------------------------------
    DOWN_LEFT = or_bits(DOWN, LEFT)  # down-left
    DOWN_RIGHT = or_bits(DOWN, RIGHT)  # down-right

    UP_LEFT = or_bits(UP, LEFT)  # up-left
    UP_RIGHT = or_bits(UP, RIGHT)  # up-right

    DOWN_LIGHT_PUNCH = or_bits(DOWN, LIGHT_PUNCH)  # crouching LP
    DOWN_HEAVY_PUNCH = or_bits(DOWN, HEAVY_PUNCH)  # crouching HP
    DOWN_LIGHT_KICK = or_bits(DOWN, LIGHT_KICK)  # crouching LK
    DOWN_MEDIUM_KICK = or_bits(DOWN, MEDIUM_KICK)  # crouching MK
    DOWN_HEAVY_KICK = or_bits(DOWN, HEAVY_KICK)  # crouching HK

    # (Add more if you want, e.g., UP+heavy punch, etc.)

    # -------------------------------------------------------
    # 4) Build the final list in a fixed order
    #    Index 'x' picks which one to return.
    # -------------------------------------------------------
    button_map = [
        NEUTRAL,  # 0
        LEFT,  # 1
        RIGHT,  # 2
        DOWN,  # 3
        UP,  # 4
        DOWN_LEFT,  # 5
        DOWN_RIGHT,  # 6
        UP_RIGHT,  # 7
        UP_LEFT,  # 8
        LIGHT_PUNCH,  # 9
        MEDIUM_PUNCH,  # 10
        HEAVY_PUNCH,  # 11
        LIGHT_KICK,  # 12
        MEDIUM_KICK,  # 13
        HEAVY_KICK,  # 14
        DOWN_LIGHT_PUNCH,  # 15
        DOWN_HEAVY_PUNCH,  # 16
        DOWN_LIGHT_KICK,  # 17
        DOWN_MEDIUM_KICK,  # 18
        DOWN_HEAVY_KICK,  # 19
    ]

    if x < 0 or x >= len(button_map):
        raise ValueError(f"x={x} is out of range [0..{len(button_map)-1}].")

    return button_map[x]


def compute_reward(prev_state, current_state, prev_score, score, step_count):
    # Extract health values
    prev_health = np.max([0, prev_state["health"]])
    prev_enemy_health = np.max([0, prev_state["enemy_health"]])
    curr_health = np.max([0, current_state["health"]])
    curr_enemy_health = np.max([0, current_state["enemy_health"]])

    player_health_lost = np.max([0, prev_health - curr_health])
    enemy_health_lost = np.max([0, prev_enemy_health - curr_enemy_health])
    delta_score = score - prev_score

    # Parameters
    alpha = 15.0  # Positive reward scaling
    beta = 10.0  # Negative penalty scaling
    S = 0.3  # Positive score scaling
    T = 0.05  # Time penalty scaling

    # Base rewards
    reward_hit = alpha * enemy_health_lost
    reward_damage = -beta * player_health_lost

    # reward score
    reward_score = S * delta_score

    # Time penalty
    reward_time = -T * (step_count * 1 / 60)  # 60 frames per second

    # Total reward
    total_reward = reward_hit + reward_damage + reward_score + reward_time

    return total_reward


def play_game(
    episodes: int = 100,
    frame_skip: int = 1,
    game_agent: BaseAgent = DQNAgent,
    load_agent: bool = True,
    game_room: str = "StreetFighterIISpecialChampionEdition-Genesis",
    render_mode: str = "rgb_array",
    record: str = "./recordings/",
):
    """
    Main loop to train a DQN agent with frame stacking.

    Parameters:
    - episodes (int): Number of episodes to play.
    - frame_skip (int): Number of frames to skip (action repeats).
    - game_agent (BaseAgent): The agent class to instantiate.
    - game_room (str): The game ROM to load.
    - render_mode (str): Rendering mode for the environment.
    - record (str): Path to save recordings.
    """
    console.print("[bold green]Starting game ...[/bold green]")
    console.print(f"\tGame: {game_room}")
    console.print(f"\tRender_mode: {render_mode}")
    console.print(f"\tRecord path: {record}")

    env = retro.make(game=game_room, render_mode=render_mode, record=record)
    initial_frame, info = env.reset()
    score = 0

    # TODO: Improve the intial state.
    initial_info = {
        "health": 176,
        "enemy_health": 176,
        "matches_won": 0,
        "enemy_matches_won": 0,
    }
    info = initial_info
    neutral_action = game_input_map(0)

    # Convert initial frame to grayscale
    initial_gray = rgb2gray_luminance(initial_frame)

    # Initialize frame buffer with 4 copies of the initial grayscale frame
    frame_stack_size = 4
    frame_buffer = deque(
        [initial_gray for _ in range(frame_stack_size)], maxlen=frame_stack_size
    )

    # Stack the initial frames
    state = stack_frames_grayscale(frame_buffer)
    print(f"state shape: {state.shape}")

    console.print("[bold green]Creating agent...[/bold green]")
    # Assuming action_dim is the number of possible discrete actions
    action_dim = 20
    agent = game_agent(
        action_dim=action_dim, state_shape=(frame_stack_size, 200, 256)
    )  # 4 grayscale frames
    console.print("[bold green]Agent created![/bold green]")

    console.print("[bold green]Game started![/bold green]")

    episode_count = 0
    total_reward = 0
    total_score = 0

    if load_agent:
        agent.load()

    while episode_count < episodes:
        done = False
        truncated = False
        episode_reward = 0
        episode_score = 0
        step_count = 0

        while not done and not truncated:
            # 1. Select action based on current state
            agent_action = agent.act(
                state.numpy()
            )  # Convert tensor to NumPy for the agent

            # 2. Apply the action for 'frame_skip' frames
            for _ in range(frame_skip):
                next_frame, next_score, done, truncated, next_info = env.step(
                    game_input_map(agent_action)
                )

                # 3. Convert next frame to grayscale
                next_gray = rgb2gray_luminance(next_frame)

                # 4. Update frame buffer
                frame_buffer.append(next_gray)

                # 5. Stack frames to form the next state
                next_state = stack_frames_grayscale(frame_buffer)

                # 6. Calculate reward
                reward = compute_reward(info, next_info, score, next_score, step_count)

                # 7. Remember the transition
                agent.remember(
                    state.numpy(),
                    agent_action,
                    reward,
                    next_state.numpy(),
                    done or truncated,
                )

                # 8. Update agent
                agent.replay()
                agent.decay_epsilon()

                # 9. Move to next state
                delta_win = next_info["matches_won"] - info["matches_won"]
                delta_lost = next_info["enemy_matches_won"] - info["enemy_matches_won"]

                state = next_state
                score = next_score
                info = next_info

                # 10. Accumulate rewards and scores
                episode_reward += reward
                episode_score = next_info["score"]
                step_count += 1

                if delta_win > 0 or delta_lost > 0:
                    step_count = 0
                    console.print("Round over!")

                # Optionally print progress every N steps
                if step_count % 120 == 0:
                    pprint(info)
                    console.print(f"agent epsilon: {agent.epsilon}")
                    console.print(f"episode score: {episode_score}")
                    console.print(f"episode reward: {episode_reward}")
                    console.print(f"step_count: {step_count}")

                if info["matches_won"] == 2:  # skip next stage animation
                    console.print("Won match!")
                    console.print("Skipping frames ...")

                    frame_skip_counter = 0
                    while (
                        next_info["health"] != 176 and next_info["enemy_health"] != 176
                    ):
                        frame_skip_counter += 1
                        next_frame, next_score, done, truncated, next_info = env.step(
                            neutral_action
                        )

                    console.print(f"skipped frames: {frame_skip_counter}")
                    console.print("Resuming ...")

                    step_count = 0

                if (info["enemy_matches_won"] == 2) or done or truncated:
                    break

        # Episode completed
        episode_count += 1
        total_reward += episode_reward
        total_score += episode_score

        console.print(f"[bold green]Episode {episode_count} completed![/bold green]")
        console.print(f"\tEpisode Reward: {episode_reward}")
        console.print(f"\tEpisode Score: {episode_score}")

        # Reset environment and frame buffer for the next episode
        initial_frame, info = env.reset()
        initial_gray = rgb2gray_luminance(initial_frame)
        frame_buffer = deque([initial_gray for _ in range(4)], maxlen=4)
        state = stack_frames_grayscale(frame_buffer)
        info = initial_info

    console.print("[bold red]Training Finished![/bold red]")
    agent.save()
    env.close()

    console.print(f"\tTotal Episodes Played: {episode_count}")
    console.print(f"\tTotal Steps Taken: {step_count}")
    console.print(f"\tTotal Reward: {total_reward}")
    console.print(f"\tAverage Score: {total_score / episodes if episodes > 0 else 0}")
