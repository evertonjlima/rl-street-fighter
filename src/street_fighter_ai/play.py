from collections import deque

import numpy as np
import retro
from agents.base import BaseAgent
# from agents.dqn_agent import DQNAgent
from agents.dqn_recurrent_agent import RecurrentDQNAgent as DQNAgent
from preprocess import preprocess_image, stack_frames
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
    alpha = 10.0  # Positive reward scaling
    beta = 10.0  # Negative penalty scaling
    S = 0.3  # Positive score scaling
    T = 0.0  # Time penalty scaling

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
    frame_print_counter: int = 120,
    frame_stack_size: int = 4,
    frame_skip: int = 1,
    agent: BaseAgent = DQNAgent(action_dim=20, state_shape=(4, 96, 96)),
    load_agent: bool = True,
    room: str = "StreetFighterIISpecialChampionEdition-Genesis",
    render_mode: str = "rgb_array",
    record: str = "./recordings/",
):
    """
    Main loop to train a DQN agent with frame stacking.

    Parameters:
    - episodes (int): Number of episodes to play.
    - frame_print_counter (int): Interval at which to log progress during each episode.
    - frame_stack_size (int): How many frames to stack for state representation.
    - frame_skip (int): Number of frames to skip (action repeats).
    - agent (BaseAgent): The agent class to instantiate.
    - load_agent (bool): Whether to load a pre-trained model for the agent.
    - room (str): The game ROM to load.
    - render_mode (str): Rendering mode for the environment.
    - record (str): Path to save recordings.
    """

    # ----------------------
    # 1. ENVIRONMENT SETUP
    # ----------------------
    console.print("[bold green]--- ENVIRONMENT SETUP ---[/bold green]")
    console.print(f"[bold yellow]Game:[/bold yellow] {room}")
    console.print(f"[bold yellow]Render Mode:[/bold yellow] {render_mode}")
    console.print(f"[bold yellow]Record Path:[/bold yellow] {record}")

    env = retro.make(game=room, render_mode=render_mode, record=record)
    initial_frame, info = env.reset()

    # -----------------------------------------
    # 2. INITIAL SCORE/INFO/NEUTRAL ACTION SET
    # -----------------------------------------
    console.print("[bold green]--- INITIALIZING SCORE AND INFO ---[/bold green]")
    score = 0
    # TODO: Improve the initial state if needed
    initial_info = {
        "health": 176,
        "enemy_health": 176,
        "matches_won": 0,
        "enemy_matches_won": 0,
    }
    info = initial_info
    neutral_action = game_input_map(0)

    # -------------------------------
    # 3. INITIAL FRAME PREPARATION
    # -------------------------------
    console.print("[bold green]--- PREPARING INITIAL FRAME STACK ---[/bold green]")
    initial_gray = preprocess_image(initial_frame)
    frame_buffer = deque(
        [initial_gray for _ in range(frame_stack_size)], maxlen=frame_stack_size
    )
    initial_state = stack_frames(frame_buffer)
    state = initial_state
    console.print("[bold blue]Initial State Shape:[/bold blue] ", state.shape)

    # -----------------------
    # 4. START THE MAIN LOOP
    # ------------------------
    console.print("[bold green]--- STARTING GAME LOOP ---[/bold green]")
    episode_count = 0
    total_reward = 0
    total_score = 0

    while episode_count < episodes:
        console.print(
            f"[bold magenta]--- BEGIN EPISODE {episode_count + 1} ---[/bold magenta]"
        )

        # Per-episode variables
        done = False
        truncated = False
        episode_reward = 0
        episode_score = 0
        step_count = 0

        while not done and not truncated:
            # -----------------------------------------------------
            # 4.1 AGENT SELECTS ACTION BASED ON CURRENT OBSERVATION
            # -----------------------------------------------------
            agent_action = agent.act(state.numpy())  # Convert tensor to NumPy

            # ---------------------------------------
            # 4.2 APPLY ACTION FOR 'frame_skip' STEPS
            # ---------------------------------------
            for _ in range(frame_skip):
                next_frame, next_score, done, truncated, next_info = env.step(
                    game_input_map(agent_action)
                )

                # --------------------------------------------
                # 4.3 CONVERT NEXT FRAME + UPDATE FRAME BUFFER
                # --------------------------------------------
                next_gray = preprocess_image(next_frame)
                frame_buffer.append(next_gray)
                next_state = stack_frames(frame_buffer)

                # -----------------------
                # 4.4 CALCULATE REWARD
                # -----------------------
                reward = compute_reward(info, next_info, score, next_score, step_count)

                # -------------------------
                # 4.5 REMEMBER EXPERIENCE
                # -------------------------
                agent.remember(
                    state.numpy(),
                    agent_action,
                    reward,
                    next_state.numpy(),
                    done or truncated,
                )

                # -----------------------
                # 4.6 TRAIN THE AGENT
                # -----------------------
                agent.replay()
                agent.decay_epsilon()

                # -------------------------
                # 4.7 UPDATE STATE / SCORE
                # -------------------------
                delta_win = next_info["matches_won"] - info["matches_won"]
                delta_lost = next_info["enemy_matches_won"] - info["enemy_matches_won"]

                state = next_state
                score = next_score
                info = next_info

                # -----------------------------
                # 4.8 ACCUMULATE REWARDS/SCORE
                # -----------------------------
                episode_reward += reward
                episode_score = next_info.get("score", 0)
                step_count += 1

                # --------------------------------------
                # 4.9 CHECK FOR ROUND COMPLETION EVENTS
                # --------------------------------------
                if delta_win > 0 or delta_lost > 0:
                    step_count = 0
                    console.print("[bold cyan]Round over![/bold cyan]")

                # ---------------------------
                # 4.10 OPTIONAL PROGRESS LOG
                # ---------------------------
                if step_count % frame_print_counter == 0:
                    console.print("[bold blue]--- INFO DICTIONARY ---[/bold blue]")
                    pprint(info)
                    console.print(
                        f"[bold yellow]agent epsilon:[/bold yellow] {agent.epsilon}"
                    )
                    console.print(
                        f"[bold yellow]episode score:[/bold yellow] {episode_score}"
                    )
                    console.print(
                        f"[bold yellow]episode reward:[/bold yellow] {episode_reward}"
                    )
                    console.print(
                        f"[bold yellow]step_count:[/bold yellow] {step_count}"
                    )

                # 4.12 CHECK EPISODE COMPLETION
                # -----------------------------
                if (info["enemy_matches_won"] == 2) or done or truncated:
                    break

                # 4.11 FRAME SKIPPING / ROUND END
                # -------------------------------
                if info["health"] < 0 or info["enemy_health"] < 0:

                    console.print("[bold cyan]Skipping frames ...[/bold cyan]")
                    frame_skip_counter = 0

                    pprint(info)

                    # skip frames until game resets
                    while (
                        info["health"] < 0 or info["enemy_health"] < 0
                    ):
                        frame_skip_counter += 1
                        next_frame, next_score, done, truncated, info = env.step(
                            neutral_action
                        )

                    console.print(
                        f"[bold cyan]Skipped frames:[/bold cyan] {frame_skip_counter}"
                    )
                    console.print("[bold cyan]Resuming ...[/bold cyan]")

                    # reset states for next round ...
                    step_count = 0  
                    next_info = info
                    state = initial_state 





        # ----------------------------
        # 5. EPISODE COMPLETION LOGIC
        # ----------------------------
        episode_count += 1
        total_reward += episode_reward
        total_score += episode_score

        # reset agent memory buffer
        agent.reset()

        console.print(f"[bold green]Episode {episode_count} completed![/bold green]")
        console.print(f"\t[bold yellow]Episode Reward:[/bold yellow] {episode_reward}")
        console.print(f"\t[bold yellow]Episode Score:[/bold yellow]  {episode_score}")

        # -------------------------
        # 5.1 RESET FOR NEXT EPISODE
        # -------------------------
        console.print(
            "[bold green]--- RESETTING ENVIRONMENT FOR NEXT EPISODE ---[/bold green]"
        )
        initial_frame, info = env.reset()
        initial_gray = preprocess_image(initial_frame)
        frame_buffer = deque(
            [initial_gray for _ in range(frame_stack_size)], maxlen=frame_stack_size
        )
        state = stack_frames(frame_buffer)
        info = initial_info
        score = 0

    # -------------------------
    # 6. TRAINING FINISHED
    # -------------------------
    console.print("[bold red]--- TRAINING FINISHED ---[/bold red]")
    agent.save()
    env.close()

    console.print(f"[bold yellow]Total Episodes Played:[/bold yellow] {episode_count}")
    console.print(f"[bold yellow]Total Steps Taken:[/bold yellow] {step_count}")
    console.print(f"[bold yellow]Total Reward:[/bold yellow] {total_reward}")
    average_score = (total_score / episodes) if episodes > 0 else 0
    console.print(f"[bold yellow]Average Score:[/bold yellow] {average_score}")
