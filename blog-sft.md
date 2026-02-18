# Instruction Fine-Tuning a 30M Model to Generate Short Stories

*Teaching a pre-trained language model to follow prompts like "write a sad story about a lost dog."*

---

**Prompt:**
```
Write a story about: a little girl who lost her toy
With: sad ending
```

**30M SFT output:**
> Once upon a time, there was a little girl. She had a toy that was very special. It was a big, red toy. She loved it so much that she hugged it tightly.
>
> One day, the little girl lost her toy. She looked everywhere but couldn't find it. It was nowhere to be found. The little girl was very sad. She started to cry.
>
> The little girl was so sad. She asked all her friends, but none of them had the toy. Then she said, "Please, let me hug it." But it was too late. The toy was gone forever.

**125M SFT output (same prompt):**
> Once upon a time, there was a little girl who was very lucky. She had a beautiful toy box full of fun toys. One day, she wanted to show her toy box to her parents. She was so excited that she couldn't wait to get home. But when she got home, she noticed that her mom and dad were not there to show her the toy box. She was so sad and felt very unlucky. She began to cry and even more tears rolled down her face. She was so unlucky that she never got to show her parents her lucky toy box again.

The [pre-trained models](/blog/pretrain) generate coherent stories, but they can't follow instructions. If you prompt them with "write a sad story about a dog," they just continue from those words without understanding it as an instruction. This post covers how I fine-tuned those models to actually follow prompts.

## The problem with the raw dataset

The obvious dataset for instruction fine-tuning is [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) - 21.8 million instruction-response pairs for children's stories. But there's a problem: it's massively imbalanced. About 92% of the examples have happy endings. Only 8% are sad.

If you fine-tune on this data as-is, the model learns to ignore the ending instruction entirely. Ask for a sad story? You get a happy one. The model just picks the dominant pattern. The ending instruction becomes meaningless.

## Balancing the dataset

I fixed this imbalance by keping ALL of the sad examples (they're the minority class) and randomly sampled an equal number of happy ones.


| Split | Happy | Sad | Total |
|-------|-------|-----|-------|
| Train | 162,492 | 162,492 | 324,984 |
| Validation | 1,771 | 1,771 | 3,542 |

From 21.8 million examples down to 325K. It seems like a huge reduction, but the model already knows how to write stories from pre-training. It just needs to learn the mapping between instructions and story types. 

But before balancing, I had to parse the raw dataset. TinyStoriesInstruct stores examples in a weird format - multiple rows of text delimited by `<|endoftext|>` tokens, with `Summary:`, `Features:`, and `Story:` fields embedded in the text. So I wrote a streaming parser that accumulates rows into complete examples and extracts the fields.

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
            current_example = []
```

The ending type comes from the `Features` field - if it contains `BadEnding`, it's a sad story. Everything else is happy. The balanced dataset is uploaded to HuggingFace as [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced).

## Instruction formatting

Each training example needs a consistent format so the model learns the pattern. I used a simple template:

```python
def format_instruction(summary: str, ending: str) -> str:
    return f"""Write a story about: {summary}
With: {ending} ending

### Story:
"""
```

During training, the model sees the full thing - instruction + story. During inference, it only sees up to `### Story:` and generates the rest. The `### Story:` marker acts as a separator that tells the model "everything before this is the instruction, everything after is what I should generate."

A full training example looks like:
```
Write a story about: a dog who finds a magic bone
With: happy ending

### Story:
Once upon a time, there was a small dog named Buddy...
```

## Tokenization

Tokenization here is different from [pre-training](/blog/pretrain#tokenization). In pre-training, I tokenized the entire dataset as one long sequence and sliced it into fixed-length windows. For SFT, each example is independent - instruction + response concatenated and tokenized as a single sequence.

```python
class InstructionDataset(Dataset):
    def __init__(self, split="train", max_length=512):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        ds = load_dataset("0rn0/tinystories-instruct-balanced", split=split)

        self.tokenized_data = []
        for example in ds:
            full_text = example["instruction"] + example["response"]
            token_ids = self.tokenizer.encode(full_text)

            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            self.tokenized_data.append(torch.tensor(token_ids, dtype=torch.long))
```

One thing to note: the `<|endoftext|>` token is NOT added here. That's handled at batch time by the collate function. I originally had it in the dataset class, but moved it to the collate function after realizing it's more flexible that way - you can change padding behavior without re-tokenizing the entire dataset.

## Collation: padding and masking

Since each example has a different length, they need to be padded to the same length within a batch. The collate function does three things:

1. **Adds the EOT token** (`<|endoftext|>`, ID 50256) to the end of each sequence
2. **Pads** shorter sequences with the EOT token to match the longest in the batch
3. **Creates targets** (shifted by 1) and **masks extra padding** with `-100`

The key insight is which tokens get masked. The first EOT token is NOT masked - we want the model to learn when to stop generating. Only the extra padding tokens after it are masked with `-100`, which tells PyTorch's cross-entropy loss to ignore those positions.

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100):
    for item in input_ids_list:
        new_item = item.copy()
        new_item += [pad_token_id]                                    # Add EOT
        padded = new_item + [pad_token_id] * (batch_max - len(new_item))  # Pad

        inputs = torch.tensor(padded[:-1])                            # Input sequence
        targets = torch.tensor(padded[1:])                            # Shifted by 1

        # Mask extra padding (but NOT the first EOT)
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index                       # -100 = ignored
```

Here's what it looks like with real numbers. Say an example has tokens `[1, 2, 3]` and the batch max is 6:

```
After EOT:    [1, 2, 3, 50256]
After pad:    [1, 2, 3, 50256, 50256, 50256]

inputs:       [1, 2, 3, 50256, 50256]
targets:      [2, 3, 50256, 50256, 50256]
                       ↑ first EOT (NOT masked - model learns to stop)
                              ↑ extra padding (masked with -100)
                                     ↑ extra padding (masked with -100)

final targets: [2, 3, 50256, -100, -100]
```

This collate function is based on [Sebastian Raschka's approach](https://github.com/rasbt/LLMs-from-scratch) from LLMs from Scratch. I originally had a simpler version that padded with zeros and masked everything at the end, but Raschka's method of using the EOT token as the pad and selectively masking is cleaner.

## Fine-tuning

The training setup is straightforward compared to [pre-training](/blog/pretrain#training). No DDP, no multi-GPU - SFT on 325K examples is fast enough on a single GPU.

The loss function is the same cross-entropy as pre-training, but with `ignore_index=-100` so the padding positions don't contribute to the loss:

```python
def compute_loss(model_output, labels, ignore_index=-100):
    batch_size, seq_len, vocab_size = model_output.shape
    logits = model_output.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss = F.cross_entropy(logits, labels_flat, ignore_index=ignore_index)
    return loss
```

The optimizer is AdamW with a learning rate of 5e-5 and weight decay of 0.1. The learning rate is much lower than what you'd use for pre-training - you don't want to overwrite what the model already knows about language. You're nudging it toward instruction-following, not retraining it.

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        logits = model(batch["input_ids"].to(device))
        loss = compute_loss(logits, batch["labels"].to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

I fine-tuned two variants:

| | 30M SFT | 125M SFT |
|---|---------|----------|
| Base model | [gpt2-30m-tinystories](https://huggingface.co/0rn0/gpt2-30m-tinystories) | [gpt2-125m-tinystories](https://huggingface.co/0rn0/gpt2-125m-tinystories) |
| Context length | 512 | 512 |
| Embedding dim | 384 | 768 |
| Heads | 6 | 12 |
| Layers | 6 | 12 |
| Learning rate | 5e-5 | 5e-5 |
| Batch size | 8 | 8 |
| Epochs | 5 | 1 |

The pre-trained checkpoints are pulled from HuggingFace where I uploaded them after [pre-training](/blog/pretrain).

## Training on cloud GPUs

Same setup as pre-training - SSH into a Lambda Labs instance, clone the repo, and run:

```bash
ssh -i ~/Downloads/llmproject.pem ubuntu@<IP_address>
```

```bash
git clone https://github.com/AryanDeore/monday-morning-moral-sft.git
export TERM=xterm-256color
curl -LsSf https://astral.sh/uv/install.sh | sh

cd monday-morning-moral-sft && uv sync && mkdir checkpoints
```

Copy the pre-trained checkpoints from my local machine to the cloud instance:

```bash
rsync -avz -e "ssh -i ~/Downloads/llmproject.pem" \
    checkpoints/pre_trained_gpt2-30m \
    checkpoints/pre_trained_gpt2-125m \
    ubuntu@<IP_address>:~/monday-morning-moral-sft/checkpoints/
```

Then fine-tune:

```bash
uv run python finetune.py --model-size 30m --epochs 5 --batch-size 64

uv run python finetune.py --model-size 125m --epochs 2 --batch-size 64
```

After training, download the fine-tuned checkpoints back:

```bash
rsync -avz -e "ssh -i ~/Downloads/llmproject.pem" \
    ubuntu@<IP_address>:~/monday-morning-moral-sft/checkpoints/sft_30M_model/ \
    checkpoints/sft_30M_model/
```

### Things that broke

**EOT in the wrong place.** I initially appended the `<|endoftext|>` token in the `InstructionDataset` during tokenization. This meant it was baked into the stored tensors. When I wanted to experiment with different padding strategies in the collate function, I had to re-tokenize the entire dataset each time. Moving the EOT append to the collate function fixed this - tokenize once, experiment freely.

## Generating stories

At inference, the model receives the instruction prompt (everything up to `### Story:`) and generates the story autoregressively. The instruction prefix is stripped from the output so you only see the story.

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

Generation stops when the model outputs `<|endoftext|>`. This is why we don't mask the first EOT during training - the model learned that this token means "the story is done."

## Deployment

The fine-tuned model runs on a [Gradio](https://gradio.app/) web interface deployed on [Railway](https://railway.com/). At startup, it downloads the checkpoint from HuggingFace Hub, loads the model, and serves the UI. Users enter a topic, pick happy or sad, adjust the temperature, and get a story.

```python
checkpoint_path = hf_hub_download(repo_id="0rn0/gpt2-30m-tinystories-sft",
                                   filename="finetune_epoch_5.pt")
model = load_model(checkpoint_path, DEVICE)

with gr.Blocks(title="Tiny Tales GPT") as demo:
    topic = gr.Textbox(label="Generate a short story about:", ...)
    ending = gr.Radio(choices=["Happy", "Sad"], label="With ending:", value="Happy")
    temperature = gr.Slider(minimum=0.1, maximum=1.4, value=0.7, label="Temperature")
    submit_btn = gr.Button("Generate Story", variant="primary")
    output = gr.Textbox(label="Generated Story", lines=10)
```

Try it at [tinytales.aryandeore.ai](https://tinytales.aryandeore.ai/).

## What I learned

**Data quality beats data quantity.** Going from 21.8M unbalanced examples to 325K balanced ones made the model dramatically better at following the ending instruction.

**SFT is surprisingly simple.** After the complexity of pre-training (DDP, multi-GPU, 500M+ tokens, hours of training), fine-tuning felt almost anticlimactic. Same loss function, same optimizer, smaller dataset, fewer epochs. The hard work was in the data preparation and the collate function.



## What's next

- Int8 quantization for smaller model size
- Deploy quantized models to HuggingFace Spaces

Both fine-tuned models are on HuggingFace: [30M SFT](https://huggingface.co/0rn0/gpt2-30m-tinystories-sft) and [125M SFT](https://huggingface.co/0rn0/gpt2-125m-tinystories-sft).

---

*Full source code: [GitHub](https://github.com/AryanDeore/monday-morning-moral-sft) | Try the live demo: [tinytales.aryandeore.ai](https://tinytales.aryandeore.ai/)*

## References

- [LLMs from Scratch - Instruction Fine-Tuning](https://www.youtube.com/watch?v=4yNswvhPWCQ&list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11&index=7)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) - Sebastian Raschka (collate function based on his approach)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng
