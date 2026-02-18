<div align="center">

# Tiny Tales GPT - SFT

[![Demo](https://img.shields.io/badge/Live-%23FE4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://tinytales.aryandeore.ai/)
[![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/0rn0)

</div>

Instruction fine-tuned GPT-2 models that follow prompts like "write a happy story about a brave knight." This repo takes the [pre-trained TinyStories GPT models](https://github.com/AryanDeore/Tiny-Tales-GPT) and fine-tunes them on a custom balanced instruction dataset to generate stories with specific topics and ending types (happy or sad).

Two fine-tuned variants are available on HuggingFace: [30M SFT](https://huggingface.co/0rn0/gpt2-30m-tinystories-sft) and [125M SFT](https://huggingface.co/0rn0/gpt2-125m-tinystories-sft).

**Prompt:**
```
Write a story about: a little girl who lost her toy
With: sad ending
```

**125M SFT output:**
```
Once upon a time, there was a little girl who was very lucky. She had a beautiful toy box full
of fun toys. One day, she wanted to show her toy box to her parents. She was so excited that
she couldn't wait to get home. But when she got home, she noticed that her mom and dad were
not there to show her the toy box. She was so sad and felt very unlucky. She began to cry and
even more tears rolled down her face. She was so unlucky that she never got to show her parents
her lucky toy box again.
```

**30M SFT output (same prompt):**
```
Once upon a time, there was a little girl. She had a toy that was very special. It was a big,
red toy. She loved it so much that she hugged it tightly.

One day, the little girl lost her toy. She looked everywhere but couldn't find it. It was
nowhere to be found. The little girl was very sad. She started to cry.

The little girl was so sad. She asked all her friends, but none of them had the toy. Then she
said, "Please, let me hug it." But it was too late. The toy was gone forever.
```

## Table of Contents
- [Pre-trained Models](#pre-trained-models)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Step by Step Explanation](#step-by-step-explanation)
  - [Curating the Instruction Dataset](#1-curating-the-instruction-dataset)
  - [Instruction Formatting](#2-instruction-formatting)
  - [Tokenization](#3-tokenization)
  - [Collation: Padding and Masking](#4-collation-padding-and-masking)
  - [Fine-Tuning](#5-fine-tuning)
  - [Generating Stories](#6-generating-stories)
  - [Gradio Web UI](#7-gradio-web-ui)
- [Usage](#usage)
- [What's Next](#whats-next)

## Pre-trained Models

The SFT models start from pre-trained GPT-2 checkpoints trained on [TinyStoriesV2_cleaned](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned). The pre-training repo and full architecture details are at [Tiny-Tales-GPT](https://github.com/AryanDeore/Tiny-Tales-GPT).

| Config | 30M SFT | 125M SFT |
|--------|---------|----------|
| Base model | [gpt2-30m-tinystories](https://huggingface.co/0rn0/gpt2-30m-tinystories) | [gpt2-125m-tinystories](https://huggingface.co/0rn0/gpt2-125m-tinystories) |
| Context length | 512 | 512 |
| Embedding dim | 384 | 768 |
| Attention heads | 6 | 12 |
| Transformer layers | 6 | 12 |
| SFT learning rate | 5e-5 | 5e-5 |
| SFT batch size | 8 | 8 |
| SFT epochs | 5 | 1 |
| Optimizer | AdamW (weight_decay=0.1) | AdamW (weight_decay=0.1) |

## Dataset

The raw [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) dataset has 21.8M+ rows, but it's heavily imbalanced: ~92% happy endings, ~8% sad endings. A model trained on this would almost always generate happy stories regardless of the instruction.

I created a balanced version ([0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced)) by keeping ALL sad examples and randomly sampling an equal number of happy examples:

| Split | Happy | Sad | Total |
|-------|-------|-----|-------|
| Train | 162,492 | 162,492 | 324,984 |
| Validation | 1,771 | 1,771 | 3,542 |

Each example has three fields:
```python
{
    "instruction": "Write a story about: a dog who finds a magic bone\nWith: happy ending\n\n### Story:\n",
    "response": "Once upon a time, there was a small dog named Buddy...",
    "ending": "happy"
}
```

## Code Structure

```
monday-morning-moral-sft/
├── models/                          # GPT-2 architecture (from pre-training repo)
│   ├── gpt2.py                      # Main model class (PyTorchModelHubMixin)
│   ├── transformer.py               # Transformer block with residual connections
│   ├── multi_head_attention.py      # Multi-head causal self-attention
│   ├── feed_forward.py              # FFN with GELU activation
│   └── embeddings.py                # Token + positional embeddings
├── data/
│   ├── prepare_dataset.py           # Parse raw TinyStoriesInstruct + balance 50-50
│   ├── instruction_dataset.py       # InstructionDataset (tokenizes at init)
│   ├── collate.py                   # Custom collate: pad, shift, mask
│   └── dataloader.py                # DataLoader factory
├── utils/
│   ├── config.py                    # Model configs (30M and 125M)
│   └── formatting.py                # Instruction prompt template
├── finetune.py                      # Main fine-tuning script
├── generate.py                      # Story generation (topic + ending)
├── checkpoint.py                    # Checkpoint save/load utilities
├── app.py                           # Gradio web UI
└── upload_to_hf.py                  # Upload fine-tuned models to HuggingFace
```

## Step by Step Explanation

### 1. Curating the Instruction Dataset

The raw TinyStoriesInstruct dataset stores each example as multiple rows of text delimited by `<|endoftext|>`. Each example contains a `Summary:` field (the topic), a `Features:` field (which includes `BadEnding` for sad stories), and a `Story:` field. The parser streams through the raw dataset, accumulates rows into complete examples, extracts the fields, and classifies the ending type.

To balance the dataset, we keep all sad examples and randomly sample an equal number of happy ones, then shuffle and upload to HuggingFace Hub.

```python
def parsed_examples_generator(split="train"):
    raw_ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True)
    current_example = []

    for row in raw_ds:
        current_example.append(row["text"])

        if "<|endoftext|>" in row["text"]:
            full_text = "\n".join(current_example)
            # Extract Summary, Features, Story fields
            # Classify ending: "sad" if "BadEnding" in Features, else "happy"
            # Format instruction and yield
            current_example = []
```

### 2. Instruction Formatting

Each training example is formatted as an instruction prompt followed by the story. At inference time, the model receives everything up to `### Story:` and generates the continuation.

```python
def format_instruction(summary: str, ending: str) -> str:
    return f"""Write a story about: {summary}
With: {ending} ending

### Story:
"""
```

> Full training example = instruction + response. At inference, model only sees the instruction.

### 3. Tokenization

The `InstructionDataset` class loads the balanced dataset from HuggingFace and tokenizes every example at initialization using the GPT-2 TikToken encoding. Each example is the concatenation of the instruction and response, truncated to `max_length` (512 tokens). The `<|endoftext|>` token is NOT added here - that's handled by the collate function at batch time.

```python
class InstructionDataset(Dataset):
    def __init__(self, split="train", max_length=512, repo_id="0rn0/tinystories-instruct-balanced"):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        ds = load_dataset(repo_id, split=split)

        self.tokenized_data = []
        for example in ds:
            full_text = example["instruction"] + example["response"]
            token_ids = self.tokenizer.encode(full_text)

            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            self.tokenized_data.append(torch.tensor(token_ids, dtype=torch.long))
```

### 4. Collation: Padding and Masking

Since examples have different lengths, they need to be padded to the same length within a batch. The collate function (based on Raschka's approach) handles three things at batch time:

1. **Adds EOT token** (`<|endoftext|>`, ID 50256) to the end of each example
2. **Pads** shorter sequences with the EOT token to match the longest sequence in the batch
3. **Creates labels** (targets shifted by 1) and **masks extra padding** with `-100` so the loss function ignores it

The first EOT token is NOT masked - we want the model to learn when to stop generating. Only the extra padding tokens after it are masked.

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100):
    for item in input_ids_list:
        new_item = item.copy()
        new_item += [pad_token_id]                                   # Add EOT
        padded = new_item + [pad_token_id] * (batch_max - len(new_item))  # Pad

        inputs = torch.tensor(padded[:-1])                           # Input sequence
        targets = torch.tensor(padded[1:])                           # Shifted by 1

        # Mask extra padding (but NOT the first EOT)
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index                      # -100 = ignored in loss
```

> Key insight: input_ids[i] predicts labels[i]. First EOT teaches the model to stop. Extra padding is ignored.

### 5. Fine-Tuning

Fine-tuning loads the pre-trained checkpoint, freezes nothing (full fine-tuning), and trains with cross-entropy loss. The loss computation ignores positions where the label is `-100` (padding), so the model only learns from real tokens. AdamW with weight decay of 0.1 provides regularization.

```python
def compute_loss(model_output, labels, ignore_index=-100):
    batch_size, seq_len, vocab_size = model_output.shape
    logits = model_output.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits, labels_flat, ignore_index=ignore_index)
    return loss

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        logits = model(batch["input_ids"].to(device))
        loss = compute_loss(logits, batch["labels"].to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 6. Generating Stories

At inference, the `generate_story` function formats the user's topic and ending type into an instruction prompt, tokenizes it, and feeds it to the model. The model generates tokens autoregressively using temperature scaling and top-k sampling (same as the pre-training repo). Generation stops when the model produces the `<|endoftext|>` token, and the instruction prefix is stripped from the output.

```python
def generate_story(model, topic, ending, max_new_tokens=200,
                   temperature=1.0, top_k=50, eos_id=50256, device="cpu"):
    instruction_prompt = format_instruction(topic, ending)
    input_ids = text_to_token_ids(instruction_prompt, tokenizer).to(device)

    output_ids = generate(
        model=model, idx=input_ids, max_new_tokens=max_new_tokens,
        context_size=actual_context_length, temperature=temperature,
        top_k=top_k, eos_id=eos_id
    )

    full_text = token_ids_to_text(output_ids, tokenizer)
    story = full_text[len(instruction_prompt):]    # Strip instruction prefix
    return story.strip()
```

### 7. Gradio Web UI

The app downloads the fine-tuned model from HuggingFace Hub at startup and serves a Gradio interface. Users enter a topic, pick an ending type (happy/sad), adjust the temperature slider, and get a generated story. Deployed on Railway at [tinytales.aryandeore.ai](https://tinytales.aryandeore.ai/).

```python
# Download model from HF Hub at startup
checkpoint_path = hf_hub_download(repo_id="0rn0/gpt2-30m-tinystories-sft",
                                   filename="finetune_epoch_5.pt")
model = load_model(checkpoint_path, DEVICE)

# Gradio UI
with gr.Blocks(title="Tiny Tales GPT") as demo:
    topic = gr.Textbox(label="Generate a short story about:", ...)
    ending = gr.Radio(choices=["Happy", "Sad"], label="With ending:", value="Happy")
    temperature = gr.Slider(minimum=0.1, maximum=1.4, value=0.7, label="Temperature")
    submit_btn = gr.Button("Generate Story", variant="primary")
    output = gr.Textbox(label="Generated Story", lines=10)

    submit_btn.click(fn=generate, inputs=[topic, ending, temperature], outputs=output)
```

## Usage

**Generate a story (CLI):**
```bash
python generate.py --model-size 30m --topic "a brave knight" --ending happy

# Interactive mode (prompts for topic and ending)
python generate.py --model-size 125m
```

**Fine-tune a model:**
```bash
python finetune.py --model-size 30m --batch-size 8 --lr 5e-5 --epochs 5

# Quick test with limited data
python finetune.py --model-size 30m --max-tokens 100000 --epochs 1
```

**Run the Gradio app locally:**
```bash
python app.py
```

**Prepare the balanced dataset from scratch:**
```bash
python data/prepare_dataset.py
```

## What's Next

- Int8 quantization for smaller model size
- Deploy quantized models to HuggingFace Spaces

## References

- [LLMs from Scratch - Instruction Fine-Tuning](https://www.youtube.com/watch?v=4yNswvhPWCQ&list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11&index=7)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) - Sebastian Raschka (collate function based on his approach)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng

---

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aryandeore) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:aryandeore.work@gmail.com)
