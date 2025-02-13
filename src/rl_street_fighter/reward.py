"""Collection of reward functions"""

import numpy as np

def compute_reward(prev_state, current_state, prev_score, score, step_count, scale_dmg: float = 100.0, adv_dmg: float = 0.2, scale_rend: float = 100.0, scale_hend: float = 100.0):
    """
    Compute the reward for a single environment step in a Street Fighter game.
    
    Reward components:
      • Damage dealt to enemy (positive).
      • Damage received by player (negative).
      • Round outcome based on health difference (scaled by R_e).
      • Special bonus for a perfect round.
      • Tie yields zero reward.
    
    This function does not impose a preference for aggressive or defensive
    play, letting the policy figure out its own strategy.
    """
    # -------------------------------------------------------------------------
    # 1. Basic Setup & Constants
    # -------------------------------------------------------------------------
    # Health values can go below 0 in some cases; we clamp to 0 to avoid 
    # negative health confusion.
    prev_health = np.maximum(0, prev_state["health"])
    prev_enemy_health = np.maximum(0, prev_state["enemy_health"])
    curr_health = np.maximum(0, current_state["health"])
    curr_enemy_health = np.maximum(0, current_state["enemy_health"])
    
    # Time variables
    fps = 60.0
    curr_time = step_count / fps  # Convert steps to seconds (assuming 60 fps)
    max_time = 100.0  # Arbitrary match duration limit in seconds
    
    # Reward scaling constants
    alpha = scale_dmg * (1.0 - adv_dmg)
    beta = scale_dmg * (1.0 + adv_dmg)

    R_e = scale_rend                # Round end scaling
    scale_health_left = scale_hend  # Health-difference scaling
    perfect_health_threshold = 176  # Health threshold for a "perfect" round
    
    # -------------------------------------------------------------------------
    # 2. Damage-Based Rewards
    # -------------------------------------------------------------------------
    player_health_lost = np.maximum(0, prev_health - curr_health)
    enemy_health_lost = np.maximum(0, prev_enemy_health - curr_enemy_health)

    reward_hit = alpha * enemy_health_lost       # Positive for damaging enemy
    reward_damage = -beta * player_health_lost   # Negative for taking damage

    # -------------------------------------------------------------------------
    # 3. Check Round Status
    # -------------------------------------------------------------------------
    round_over = (
        current_state["health"] < 0 
        or current_state["enemy_health"] < 0 
        or curr_time > max_time
    )
    round_win = (curr_health >= 0 and curr_enemy_health < 0)
    round_tie = (curr_health < 0 and curr_enemy_health < 0)
    
    # -------------------------------------------------------------------------
    # 4. Calculate End-of-Round Rewards
    # -------------------------------------------------------------------------
    win_loss_reward = 0
    
    # If both knocked out simultaneously, treat as a tie (neutral reward).
    if round_tie:
        return 0
    
    # If round ends (win or loss), add outcome-based reward.
    if round_over:
        # Health difference scaled
        win_loss_reward += R_e * scale_health_left * (curr_health - curr_enemy_health)
        
        # Special bonus for a "perfect round" if you end the round with a high health
        # threshold and within time.
        if round_win and curr_health >= perfect_health_threshold and curr_time < max_time:
            win_loss_reward += 12400

    # -------------------------------------------------------------------------
    # 5. Aggregate Final Reward
    # -------------------------------------------------------------------------
    total_reward = reward_hit + reward_damage + win_loss_reward
    
    return total_reward



