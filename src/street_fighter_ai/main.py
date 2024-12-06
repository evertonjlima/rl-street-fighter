import typer
from play import play_game
from rich.console import Console

console = Console()
app = typer.Typer(help="AI plays Street Fighter Project")


@app.command()
def train(episodes: int = typer.Option(100, help="Number of episodes to train for")):
    """Train the AI model using Retro Gym"""
    console.print("[bold blue]Starting training script...[/bold blue]")
    play_game()


@app.command()
def collect():
    """Collect data for training using random agent"""
    console.print("[bold blue]Collecting data ... [/bold blue]")


if __name__ == "__main__":
    app()
