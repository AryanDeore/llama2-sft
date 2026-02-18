"""
Model and training configuration for Llama 2 15M.
"""


def get_config(model_size: str = "15m") -> dict:
    """
    Get configuration for a specific model size.

    Args:
        model_size: "15m"

    Returns:
        Dictionary containing all config values
    """
    configs = {
        "15m": {
            "dim": 288,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 6,
            "vocab_size": 32000,
            "multiple_of": 32,
            "norm_eps": 1e-5,
            "max_seq_len": 256,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 5e-5,
            "num_epochs": 2,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(configs.keys())}")

    return configs[model_size]
