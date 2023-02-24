import argparse
import os
import subprocess
import sys
import signal
import click
from .main import main
 
@click.group()
def cli():
    pass

@cli.command()
@click.option('--port', '-p', type=int, default=8501, help='Port number to run the app on.')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config.json file.')
def run_app(port, config):
    """Run the app based on the provided config file."""
    path_to_main = os.path.join(os.path.dirname(__file__), "main.py")
    cmd = [sys.executable, "-m", "streamlit", "run", "--server.port", str(port), path_to_main]

    if config is not None:
        cmd.extend(["--", f"--config-path={config}"])

    # Start the app in a subprocess
    app_proc = subprocess.Popen(cmd)

    try:
        # Wait for the subprocess to finish
        app_proc.wait()
    except KeyboardInterrupt:
        # If the user presses Ctrl-C, terminate the subprocess
        app_proc.send_signal(signal.SIGINT)
        app_proc.wait()

if __name__ == '__main__':
    cli()