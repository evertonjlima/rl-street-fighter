from typing import Any, Dict

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def pretty_print_info(info: Dict[str, Any]) -> None:
    """
    Pretty prints the info dictionary using rich's Table.

    Args:
        info (Dict[str, Any]): The dictionary containing game information to be printed.
    """
    table = Table(title="Game Info", show_header=True, header_style="bold magenta")
    table.add_column("Key", style="dim", justify="left")
    table.add_column("Value", justify="right")

    for key, value in info.items():
        table.add_row(str(key), str(value))

    console.print(table)


def rgb_to_grayscale(observation: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Args:
        observation (np.ndarray): The RGB image with shape (H, W, 3).

    Returns:
        np.ndarray: The grayscale image with shape (H, W).
    """
    return np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])  # Luminance formula
