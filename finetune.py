"""Instruction fine-tuning script for Llama 2 15M model.

Fine-tunes a pretrained Llama 2 model on instruction-following tasks
using the TinyStories instruction dataset with 50-50 happy/sad split.
"""

import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from data.dataloader import create_dataloader
from models.llama2 import Transformer
from utils.config import get_config


def load_pretrained_model(device: str = "cpu") -> Transformer:
    """Load pretrained Llama 2 15M model from HuggingFace Hub.

    Args:
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Transformer instance with loaded weights
    """
    print(f"Loading pretrained model from HuggingFace Hub...")

    # Load from HF Hub
    model = Transformer.from_pretrained("0rn0/llama2-15m-tinystories")
    model.to(device)
    model.eval()

    print(
        f"✓ Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model


def compute_loss(model_output, labels, ignore_index: int = -100) -> torch.Tensor:
    """Compute language modeling loss.

    Args:
        model_output: Model logits (batch_size, seq_len, vocab_size)
        labels: Target token IDs (batch_size, seq_len), with -100 for ignored positions
        ignore_index: Token ID to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len, vocab_size = model_output.shape

    # Flatten batch and sequence dimensions
    logits = model_output.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels_flat, ignore_index=ignore_index)
    return loss


def train_epoch(
    model: Transformer,
    train_loader,
    optimizer: AdamW,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: GPT model to train
        train_loader: Training data loader
        optimizer: Adam optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch in pbar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)

        # Compute loss
        loss = compute_loss(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: Transformer,
    val_loader,
    device: str,
) -> float:
    """Evaluate model on validation set.

    Args:
        model: GPT model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = compute_loss(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: Transformer, optimizer: AdamW, epoch: int, loss: float, checkpoint_dir: str
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"finetune_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    print(f"✓ Saved checkpoint: {checkpoint_path}")


def main(
    batch_size: int = 64,
    learning_rate: float = 5e-5,
    num_epochs: int = 2,
    max_length: int = 256,
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """Main training script for Llama 2 15M instruction fine-tuning.

    Args:
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of epochs to train
        max_length: Maximum sequence length
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ("cpu" or "cuda")
        max_tokens: Maximum tokens to train on (optional)
    """
    # Set defaults
    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints/sft_15M_model"

    # Setup
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
    elif device == "mps":
        print("Using Metal Performance Shaders (MPS) on Mac")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num epochs: {num_epochs}")
    print(f"Max length: {max_length}")
    print(f"Max tokens: {max_tokens if max_tokens is not None else 'All'}")
    print("-" * 70)

    # Load model
    model = load_pretrained_model(device=device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        split="train",
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )
    val_loader = create_dataloader(
        split="validation",
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")

    # Calculate total tokens
    total_tokens = len(train_loader) * batch_size * max_length

    # Limit training data if max_tokens specified
    if max_tokens is not None:
        max_batches = max(1, max_tokens // (batch_size * max_length))
        train_loader = list(train_loader)[:max_batches]
        total_tokens = len(train_loader) * batch_size * max_length
        print(f"  ⚠ Limited to {max_batches} batches ({total_tokens:,} tokens)")

    print(f"  Total training tokens: {total_tokens:,}")
    print(f"  Tokens per epoch: {total_tokens:,}")
    print(f"  Total tokens (all epochs): {total_tokens * num_epochs:,}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print("  Weight decay: 0.1")
    print("-" * 70)

    # Print model config
    config = get_config("llama2-15m")
    print("\nModel Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 70)

    # Training loop
    print(f"\nStarting training on {device}...")
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ Best val loss improved to {val_loss:.4f}")

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Time elapsed: {hours}h {minutes}m {seconds}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Llama 2 15M on instruction tasks")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--max-length", type=int, default=256, help="Max sequence length"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="Device (cpu, cuda, or mps)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to train on (useful for testing)",
    )

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_tokens=args.max_tokens,
    )
