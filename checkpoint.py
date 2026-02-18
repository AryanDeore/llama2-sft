"""
Checkpoint utilities for saving and loading Llama 2 models.
"""

import os
import torch
from models.llama2 import Transformer


def save_checkpoint(model, epoch, optimizer=None, config_name="llama2-15m", checkpoint_dir="checkpoints"):
    """
    Save model checkpoint with metadata and optimizer state.

    Args:
        model: Llama 2 model
        epoch: Current epoch number
        optimizer: Optimizer (optional, for resuming training)
        config_name: Name of config (e.g., "llama2-15m"). Used to create subdirectory.
        checkpoint_dir: Base directory to save checkpoints
    """
    subdir = os.path.join(checkpoint_dir, config_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    filepath = os.path.join(subdir, f"finetune_epoch_{epoch}.pt")

    # Save checkpoint with metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }

    # Save optimizer state if provided
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Epoch {epoch}: Checkpoint saved at {filepath}")


def load_model(filepath, device):
    """
    Load model from checkpoint.

    Args:
        filepath: Path to model file
        device: Device to load on

    Returns:
        Llama 2 model in eval mode
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Handle checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint (no metadata)")

    # Load from HF Hub and apply state dict
    model = Transformer.from_pretrained("0rn0/llama2-15m-tinystories")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {filepath}\n")
    return model


def load_checkpoint(filepath, device):
    """
    Load checkpoint for resuming training. Returns model, optimizer state, and epoch.

    Args:
        filepath: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (model, optimizer_state_dict, epoch) or (model, None, epoch) if optimizer state not available
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Extract model state
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", 1)
    else:
        state_dict = checkpoint
        epoch = 1
        print("Loaded legacy checkpoint (no metadata)")

    # Load from HF Hub and apply state dict
    model = Transformer.from_pretrained("0rn0/llama2-15m-tinystories")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Extract optimizer state if available
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer_state_dict, epoch
