"""Custom collate function for instruction fine-tuning batches."""

from typing import Dict, List, Optional

import torch


def custom_collate_fn(
    batch: List[Dict],
    pad_token_id: int = 2,
    ignore_index: int = -100,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu"
) -> Dict:
    """Collate batch of tokenized examples (Raschka approach).

    Adds EOS token at batch time, pads with EOS tokens, and masks extra padding.
    This approach is more flexible than adding EOS in the dataset.

    Args:
        batch: List of dicts with 'input_ids' key (torch.Tensor)
        pad_token_id: Token ID for padding (2 = Llama EOS)
        ignore_index: Token ID to use for masking in loss (-100)
        allowed_max_length: Optional max sequence length (truncate if longer)
        device: Device to place tensors on

    Returns:
        Dict with:
            - input_ids: padded token IDs (batch_size, seq_len)
            - attention_mask: mask indicating real vs padding tokens (batch_size, seq_len)
            - labels: targets for training, shifted by 1 (batch_size, seq_len)
    """
    # Extract input_ids from batch
    input_ids_list = [ex["input_ids"].tolist() if isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"] for ex in batch]

    # Find the longest sequence in the batch, accounting for EOT token we'll add
    batch_max_length = max(len(item) + 1 for item in input_ids_list)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst, attention_masks_lst = [], [], []

    for item in input_ids_list:
        # Make a copy and add EOT token
        new_item = item.copy() if isinstance(item, list) else item.tolist()
        new_item += [pad_token_id]  # Add EOT token

        # Pad sequences to batch_max_length with pad_token_id
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Create inputs and targets
        inputs = torch.tensor(padded[:-1], dtype=torch.long)  # Exclude last token
        targets = torch.tensor(padded[1:], dtype=torch.long)   # Shift by 1

        # Create attention mask (1 for real+EOT, 0 for padding)
        attention_mask = torch.ones(len(inputs), dtype=torch.long)
        original_length = len(item) + 1  # +1 for EOT we added
        if original_length < len(padded):
            attention_mask[original_length:] = 0

        # Mask extra padding tokens in targets (but NOT the first EOT)
        # Replace all but the first padding token with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to allowed_max_length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            attention_mask = attention_mask[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
        attention_masks_lst.append(attention_mask)

    # Stack into batch tensors and move to device
    input_ids = torch.stack(inputs_lst).to(device)
    labels = torch.stack(targets_lst).to(device)
    attention_mask = torch.stack(attention_masks_lst).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    """Test the collate function (Raschka approach)."""

    print("=" * 70)
    print("Custom Collate Function Test (Raschka Approach)")
    print("=" * 70)

    # Create mock batch with different lengths (NO EOT)
    batch = [
        {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
        {"input_ids": torch.tensor([6, 7, 8])},
        {"input_ids": torch.tensor([9, 10, 11, 12, 13, 14])},
    ]

    print("\nInput batch (different lengths, NO EOT):")
    for i, ex in enumerate(batch):
        print(f"  Example {i}: {ex['input_ids'].tolist()}")

    # Collate (EOT = 50256)
    collated = custom_collate_fn(batch, pad_token_id=50256)

    print(f"\nCollated batch (EOT=50256 added, max_length={collated['input_ids'].shape[1]}):")
    print(f"\ninput_ids shape: {collated['input_ids'].shape}")
    print(f"{collated['input_ids']}")

    print(f"\nattention_mask shape: {collated['attention_mask'].shape}")
    print(f"{collated['attention_mask']}")

    print(f"\nlabels shape: {collated['labels'].shape}")
    print(f"{collated['labels']}")

    # Verify correctness
    print("\n" + "=" * 70)
    print("Verification (Raschka Approach)")
    print("=" * 70)

    print("\nExample 0 analysis:")
    print(f"  Original (no EOT): {batch[0]['input_ids'].tolist()}")
    print(f"  After EOT added: {batch[0]['input_ids'].tolist() + [50256]}")
    print(f"  Padded input_ids: {collated['input_ids'][0].tolist()}")
    print(f"  Attention mask: {collated['attention_mask'][0].tolist()}")
    print(f"  Labels (targets): {collated['labels'][0].tolist()}")
    print("  Explanation:")
    print("    - input_ids[0]=1 → labels[0]=2 (next token)")
    print("    - input_ids[4]=5 → labels[4]=50256 (next token is EOT)")
    print("    - input_ids[5]=50256 → labels[5]=-100 (extra padding, ignored)")
    print("    - Key: First EOT (position 5) is NOT masked, teaches model to predict EOT")
    print("    - Extra padding after EOT is masked with -100")

    print("\n✓ Test complete")
