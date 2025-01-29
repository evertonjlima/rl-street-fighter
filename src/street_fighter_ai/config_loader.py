from pathlib import Path
from typing import Tuple, Optional

import yaml
from pydantic import BaseModel


class TrainPlayGameSettings(BaseModel):
    episodes: int
    frame_stack_size: int  # Must match agent state_shape
    frame_skip: int
    room: str
    render_mode: str
    record: Path


class AgentSettings(BaseModel):
    action_dim: int
    state_shape: Tuple[int, int, int]  # Must match frame stack
    gamma: float
    lr: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    target_update_freq: int
    replay_capacity: int


class Config(BaseModel):
    in_agent_filepath: Optional[Path] = None
    out_agent_filepath: Path
    train_play_game_settings: TrainPlayGameSettings
    agent_settings: AgentSettings


# Load configuration from a YAML file
def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.load(file, Loader=yaml.Loader)
    return Config(**config_dict)


if __name__ == "__main__":
    config = load_config("config.yaml")
    print(config.json(indent=2))
