"""DataLoader creation utilities for instruction fine-tuning."""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.collate import custom_collate_fn
from data.instruction_dataset import InstructionDataset


def create_dataloader(
    split: str = "train",
    batch_size: int = 8,
    max_examples: Optional[int] = None,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for instruction fine-tuning.

    Args:
        split: "train" or "validation"
        batch_size: Number of examples per batch
        max_examples: Optional limit on number of examples to load (for testing)
        max_length: Maximum sequence length (truncate if longer)
        shuffle: Whether to shuffle data (True for train, False for val)
        num_workers: Number of worker processes for data loading (0 = main process)

    Returns:
        DataLoader with batched, padded, and masked examples
    """
    # Load dataset
    dataset = InstructionDataset(split=split, max_length=max_length)

    # Limit examples if specified (for quick testing)
    if max_examples:
        indices = list(range(min(max_examples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    # Wrap in DataLoader (handles batching, shuffling, collation)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (split == "train"),  # Shuffle only training data
        collate_fn=custom_collate_fn,
        drop_last=True,  # Drop incomplete batches
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    print("=" * 70)
    print("DataLoader Test")
    print("=" * 70)

    # Create dataloaders
    print("\nCreating train dataloader (batch_size=8, max_length=256)...")
    train_dl = create_dataloader(split="train", batch_size=8, max_length=256)

    print("Creating validation dataloader (batch_size=8, max_length=256)...")
    val_dl = create_dataloader(split="validation", batch_size=8, max_length=256, shuffle=False)

    print("\nDataLoader sizes:")
    print(f"  Train batches: {len(train_dl):,}")
    print(f"  Val batches: {len(val_dl):,}")

    # Test with small dataset
    print("\n" + "=" * 70)
    print("Testing with 100 examples")
    print("=" * 70)

    train_dl_small = create_dataloader(split="train", batch_size=8, max_examples=100)
    print(f"  Train batches (100 examples): {len(train_dl_small)}")

    # Get first batch
    print("\nFirst batch structure:")
    for batch in train_dl_small:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"    (batch_size=8, seq_len={batch['input_ids'].shape[1]})")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")

        print("\nFirst example in batch:")
        print(f"  input_ids: {batch['input_ids'][0][:20].tolist()}... (first 20 tokens)")
        print(f"  attention_mask: {batch['attention_mask'][0].tolist()}")
        print(f"  labels: {batch['labels'][0][:20].tolist()}... (first 20 tokens)")
        break

    print("\n✓ Test complete")

