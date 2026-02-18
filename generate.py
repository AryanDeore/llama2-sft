"""
Generate stories using fine-tuned instruction-following Llama 2 model.

Supports:
- Instruction formatting (topic + ending type)
- Temperature scaling (control randomness)
- Top-k sampling (sample from top k likely tokens)
- EOS token stopping condition
- Throughput measurement (tokens/sec)
"""

import argparse
import os
import time

import torch
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

from checkpoint import load_model
from utils.formatting import format_instruction

# HF repo that stores the tokenizer
_TOKENIZER_REPO = os.environ.get("HF_REPO_ID", "0rn0/llama2-15m-tinystories-sft")


def _get_tokenizer_path():
    """Return local path to tokenizer.model, downloading from HF Hub if needed."""
    local_path = "tokenizer.model"
    if os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=_TOKENIZER_REPO, filename="tokenizer.model")


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs with batch dimension."""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text."""
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(
    model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None
):
    """
    Generate tokens autoregressively.

    Args:
        model: Llama 2 model in eval mode
        idx: Starting token indices [batch_size, seq_len]
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length for model
        temperature: Sampling temperature (>1 = more random, <1 = more greedy, 0 = greedy)
        top_k: If not None, only sample from top k tokens
        eos_id: End-of-sequence token ID. Stop if generated. If None, generate full length.

    Returns:
        Generated token IDs [batch_size, seq_len + max_new_tokens]
    """

    for _ in range(max_new_tokens):
        # Crop to context size (sliding window)
        idx_cond = idx[:, -context_size:]

        # Forward pass
        with torch.no_grad():
            logits = model(idx_cond)  # [batch, seq_len, vocab_size]

        # Get only last position logits
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Numerical stability: subtract max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        else:
            # Greedy: take argmax
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]

        # Check for EOS token
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_story(
    model,
    topic,
    ending,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    eos_id=2,
    device="cpu",
):
    """
    Generate a story given a topic and ending type.

    Args:
        model: Fine-tuned Llama 2 model in eval mode
        topic: Story topic/summary
        ending: Ending type ("happy" or "sad")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0=greedy, 1=neutral, >1=random)
        top_k: Only sample from top k tokens (None to disable)
        eos_id: End-of-sequence token ID (2 for Llama EOS)
        device: Device to generate on

    Returns:
        Generated story text
    """
    # Get tokenizer
    sp = SentencePieceProcessor(model_file=_get_tokenizer_path())

    # Format instruction prompt (without story)
    instruction_prompt = format_instruction(topic, ending)

    # Get actual context length from model
    actual_context_length = model.max_seq_len

    # Convert prompt to tokens
    input_ids = text_to_token_ids(instruction_prompt, sp).to(device)

    # Generate
    output_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=actual_context_length,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )

    # Decode and return story (excluding the instruction part)
    full_text = token_ids_to_text(output_ids, sp)
    story = full_text[len(instruction_prompt) :]  # Remove instruction prefix

    return story.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instruction-following stories"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sft_15M_model/finetune_epoch_1.pt",
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Story topic (interactive prompt if not provided)",
    )
    parser.add_argument(
        "--ending",
        type=str,
        choices=["happy", "sad"],
        default=None,
        help="Ending type: happy or sad (interactive prompt if not provided)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0=greedy, 1=neutral, >1=more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (None to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Interactive prompts if topic or ending not provided
    if args.topic is None:
        args.topic = input("Write a short story about: ").strip()
        if not args.topic:
            args.topic = "a brave knight on an adventure"

    if args.ending is None:
        ending_choice = input("With: (happy/sad) ending: ").strip().lower()
        if ending_choice not in ["happy", "sad"]:
            ending_choice = "happy"
        args.ending = ending_choice

    # Setup device
    if args.device:
        device = args.device
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    print(f"Using device: {device}\n")

    # Load fine-tuned model
    try:
        model = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Fine-tune the model first using: python finetune.py")
        exit(1)

    # Generation parameters
    print("Generation config:")
    print(f"  Topic: '{args.topic}'")
    print(f"  Ending: {args.ending}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Checkpoint: {args.checkpoint}\n")

    # Generate story
    start_time = time.time()
    story = generate_story(
        model=model,
        topic=args.topic,
        ending=args.ending,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    tokens_per_sec = args.max_tokens / elapsed_time if elapsed_time > 0 else 0

    print("Generated story:")
    print(f"\n{story}\n")

    print("Performance:")
    print(f"  Generated {args.max_tokens} tokens in {elapsed_time:.2f} seconds")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
