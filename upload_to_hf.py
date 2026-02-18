"""
Upload SFT Llama 2 model to Hugging Face Hub.

Uses push_to_hub() (via PyTorchModelHubMixin) to upload model.safetensors + config.json,
which enables download tracking and proper model loading on HF.

Usage:
    python upload_to_hf.py

Requires: huggingface-cli login (run beforehand)
"""

from checkpoint import load_model
from huggingface_hub import HfApi


def upload_model(checkpoint_path, repo_id, model_card_path):
    """Load a checkpoint and push model + model card to HF Hub."""
    model = load_model(checkpoint_path, device="cpu")

    print(f"\nPushing model to {repo_id}...")
    model.push_to_hub(repo_id)
    print(f"Model weights + config.json pushed to {repo_id}")

    # Upload model card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"Model card uploaded to {repo_id}")


if __name__ == "__main__":
    # Llama 2 15M SFT
    upload_model(
        checkpoint_path="checkpoints/sft_15M_model/finetune_epoch_3.pt",
        repo_id="0rn0/llama2-15m-tinystories-sft",
        model_card_path="model_cards/llama2-15m-sft-README.md",
    )

    print("\nDone! Llama 2 SFT model uploaded to Hugging Face.")
