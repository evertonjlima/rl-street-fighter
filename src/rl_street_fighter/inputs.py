"""Functions to manage environment inputs"""

from rich.console import Console
console = Console()

def generate_move_map():
    """
    Generate a move map (a list of 12-bit vectors) that covers all valid combinations
    of direction and attack inputs.
    
    Directions:
      - We allow horizontal: neutral, left, right.
      - We allow vertical: neutral, up, down.
      (That yields 3 x 3 = 9 directional combinations, which include the four diagonals.)
    
    Attacks:
      - We assume 6 attack buttons. Rather than a fixed set of single-button presses,
        we allow every combination of these buttons (2^6 = 64 possibilities).
      - We assign the 6 attack bits to specific positions in the 12-bit vector:
          • Medium Kick → index 0  
          • Light Kick  → index 1  
          • Heavy Kick  → index 8  
          • Medium Punch→ index 9  
          • Light Punch → index 10  
          • Heavy Punch → index 11

    The direction buttons use indices:
          • Up    → index 4  
          • Down  → index 5  
          • Right → index 6  
          • Left  → index 7

    The remaining bits (indices 2 and 3) remain 0.

    Total moves = 9 (directions) * 64 (attack combinations) = 576.
    """
    moves = []
    # Loop over directional choices.
    # For horizontal: 0 = neutral, 1 = left, 2 = right.
    # For vertical:   0 = neutral, 1 = up,   2 = down.
    for hor in range(3):
        for ver in range(3):
            # Build the direction vector (12 bits, all zeros except in indices 4-7).
            dir_vector = [0] * 12
            if ver == 1:      # up
                dir_vector[4] = 1
            elif ver == 2:    # down
                dir_vector[5] = 1
            if hor == 1:      # left
                dir_vector[7] = 1
            elif hor == 2:    # right
                dir_vector[6] = 1
            
            # For the attack part, loop over all 64 combinations.
            # We represent each attack combination as a 6-bit number.
            for attack_val in range(64):
                # Convert the attack value to 6 bits.
                # We define the order: bit0 → Medium Kick, bit1 → Light Kick,
                # bit2 → Heavy Kick, bit3 → Medium Punch, bit4 → Light Punch, bit5 → Heavy Punch.
                attack_bits = [(attack_val >> i) & 1 for i in range(6)]
                
                # Build the attack vector: 12 bits, with the 6 bits placed in the chosen positions.
                attack_vector = [0] * 12
                attack_vector[0] = attack_bits[0]  # Medium Kick
                attack_vector[1] = attack_bits[1]  # Light Kick
                attack_vector[8] = attack_bits[2]  # Heavy Kick
                attack_vector[9] = attack_bits[3]  # Medium Punch
                attack_vector[10] = attack_bits[4] # Light Punch
                attack_vector[11] = attack_bits[5] # Heavy Punch
                
                # Combine the direction and attack vectors.
                # (They use disjoint indices, so we can simply take the elementwise maximum.)
                combined = [max(d, a) for d, a in zip(dir_vector, attack_vector)]
                moves.append(combined)
    return moves

# Create the global MOVE_MAP containing all 576 moves.
MOVE_MAP = generate_move_map()

def input_to_bit(x: int):
    """
    Given an index x (0 <= x < len(MOVE_MAP)), return the corresponding 12-bit move.
    This function is used during live mode: the network outputs a discrete index,
    and game_input_map(x) returns the corresponding 12-bit array.
    """
    if x < 0 or x >= len(MOVE_MAP):
        raise ValueError(f"x={x} is out of range [0..{len(MOVE_MAP)-1}].")
    return MOVE_MAP[x]

def bit_to_input(bit_array):
    """
    Given a 12-bit vector (bit_array), return its index in MOVE_MAP.
    This can be used to convert a raw 12-bit input (from a movie, for example)
    back into the discrete action index.
    """
    try:
        return MOVE_MAP.index(bit_array)
    except ValueError:
        # If the bit array is not found, default to the neutral action (index 0).
        console.print("[bold red]Unable to map bit_array to input, defaulting to neural action[/bold red]")
        console.print(f"[bold red]unknown bit_array: {bit_array}[/bold red]")
        return 0


