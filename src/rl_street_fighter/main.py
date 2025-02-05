import pandas as pd
import typer
from agents.dqn_recurrent_agent import RecurrentDQNAgent
from config_loader import load_config
from play import play_game
from rich.console import Console
from utils import pretty_print_info as pprint

console = Console()
app = typer.Typer(help="AI plays Street Fighter Project")


@app.command()
def train(
    config_path: str = typer.Option(
        "./config/config.yaml", help="Path of configuration file with parameters"
    )
):
    """Train the AI model using Retro Gym"""
    console.print("[bold blue]Starting training script...[/bold blue]")

    console.print("[bold blue]Loading configuration...[/bold blue]")
    console.print(f"Configuration file path: {config_path}")
    cfg = load_config(config_path)

    console.print("[bold blue]Train Configuration Settings: [/bold blue]")
    pprint(dict(cfg.train_play_game_settings))

    console.print("[bold blue]Agent Configuration Settings: [/bold blue]")
    pprint(dict(cfg.agent_settings))

    console.print("[bold blue]Instantiating agent ... [/bold blue]")
    agent = RecurrentDQNAgent(**dict(cfg.agent_settings))

    if cfg.in_agent_filepath:
        console.print("[bold red]\tFound agent path in configuration![/bold red]")
        console.print(
            f"[bold red]\tLoading agent from path: {cfg.in_agent_filepath} [/bold red]"
        )
        agent.load(filename=cfg.in_agent_filepath)

    console.print("[bold blue]Starting game training session... [/bold blue]")
    results = play_game(agent=agent, **dict(cfg.train_play_game_settings))
    console.print("[bold blue]Exiting training session... [/bold blue]")

    console.print("[bold blue]Saving agent...")
    agent.save(filename=cfg.out_agent_filepath)
    console.print(f"[bold blue]Agent saved: {cfg.out_agent_filepath}")

    console.print("[bold blue]Saving results...")
    pd.DataFrame(results).to_csv(cfg.out_results_filepath)
    console.print(f"[bold blue]Results saved: {cfg.out_results_filepath}")

    console.print("[bold blue]Done![/bold blue]")


@app.command()
def collect():
    pass


if __name__ == "__main__":
    app()
