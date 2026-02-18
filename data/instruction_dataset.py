"""InstructionDataset: Loads curated dataset and tokenizes at initialization."""

from typing import Any, Dict

import torch
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning.

    Loads curated TinyStories-Instruct dataset and tokenizes each example
    at initialization. Concatenates instruction + response.
    """

    def __init__(self, split: str = "train", max_length: int = 256, repo_id: str = "0rn0/tinystories-instruct-balanced"):
        """Initialize dataset.

        Args:
            split: "train" or "validation"
            max_length: Maximum token sequence length (truncate if longer)
            repo_id: HuggingFace dataset repository ID
        """
        self.max_length = max_length
        self.split = split

        # Load tokenizer (Llama 2 sentencepiece, 32k vocab)
        self.tokenizer = SentencePieceProcessor(model_file="tokenizer.model")

        # Load dataset from HuggingFace
        print(f"Loading {split} split from {repo_id}...")
        ds = load_dataset(repo_id, split=split)

        # Tokenize all examples at initialization
        print(f"Tokenizing {len(ds)} examples...")
        self.tokenized_data = []

        for i, example in enumerate(ds):
            if i % 10000 == 0:
                print(f"  Tokenized {i:,} / {len(ds):,}")

            # Concatenate instruction + response
            # EOS will be added in the collate function
            example_dict: Dict[str, Any] = dict(example)
            full_text = example_dict["instruction"] + example_dict["response"]

            # Tokenize
            token_ids = self.tokenizer.encode(full_text)

            # Truncate to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            self.tokenized_data.append(torch.tensor(token_ids, dtype=torch.long))

        print(f"Loaded {len(self.tokenized_data)} examples")

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.tokenized_data[idx]
        }
