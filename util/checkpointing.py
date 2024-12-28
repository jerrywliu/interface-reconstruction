import os
import pickle
import time
import yaml


class Checkpoint:
    """Checkpoint manager for saving/loading simulation states and configs."""

    def __init__(self, output_dir, config=None):
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save config if provided during initialization
        if config:
            self.save_config(config)

    def save_config(self, config):
        """Save configuration for this run."""
        config_path = os.path.join(self.checkpoint_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save(self, state, iteration):
        """Save simulation state with iteration number."""
        filename = f"checkpoint_iter{iteration}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint_data = {
            "state": state,
            "iteration": iteration,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        return filepath

    def load_latest(self):
        """Load most recent checkpoint if it exists."""
        checkpoint_files = self._get_checkpoint_files()

        if not checkpoint_files:
            return None

        # Get most recent checkpoint
        latest_file = max(
            checkpoint_files, key=lambda f: int(f.split("iter")[-1].split(".")[0])
        )
        filepath = os.path.join(self.checkpoint_dir, latest_file)

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def load_iteration(self, iteration):
        """Load specific iteration if it exists."""
        filename = f"checkpoint_iter{iteration}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def get_iterations(self):
        """Get list of available checkpoint iterations."""
        checkpoint_files = self._get_checkpoint_files()
        return sorted(
            [int(f.split("iter")[-1].split(".")[0]) for f in checkpoint_files]
        )

    def load_config(self):
        """Load configuration if it exists."""
        config_path = os.path.join(self.checkpoint_dir, "config.yaml")
        if not os.path.exists(config_path):
            return None

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_checkpoint_files(self):
        """Helper to get list of checkpoint files."""
        files = os.listdir(self.checkpoint_dir)
        return [f for f in files if f.startswith("checkpoint_") and f.endswith(".pkl")]
