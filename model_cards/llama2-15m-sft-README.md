---
pipeline_tag: text-generation
library_name: pytorch
language:
  - en
license: mit
datasets:
  - 0rn0/tinystories-instruct-balanced
tags:
  - llama2
  - tinystories
  - instruction-tuning
  - sft
  - causal-lm
  - story-generation
widget:
  - text: "Write a story about: a little girl and her dog at the park\nWith: happy ending\n\n### Story:\n"
    example_title: "Happy ending"
    output:
      text: "Once upon a time there was a little girl and her loyal dog. They were best friends and did everything together. One day they went to the park and they were having lots of fun.\nSuddenly, the girl saw a big seat. She wanted to sit on it, so she asked her loyal dog, \"Can I sit on the seat?\" Her dog barked happily and the girl hopped on the seat.\nThey were both very happy and they spent the whole day playing and laughing together. They even saw some other children playing in the park.\nAt the end of the day, the girl and her loyal dog went home. The girl was sure that her loyal dog was always there to protect her and make her feel safe. She hugged him and said, \"I love you, my loyal dog!\""
  - text: "Write a story about: a boy who lost his favorite toy\nWith: sad ending\n\n### Story:\n"
    example_title: "Sad ending"
    output:
      text: "The boy said, \"Yes, I understand. I lost my toy and I can't find it.\" The man said, \"Don't worry, I'll help you find it.\"\nThey looked and looked for the toy, but they could not find it. The boy was very sad. The man said, \"I'm sorry, I can't find your toy.\" The boy went home with a sad face, and the man went back to his house with a bad feeling."

# Llama 2 15M — TinyStories SFT

## Model Details
- **Architecture**: Llama 2 (custom implementation)
- **Parameters**: ~15.2M
- **Context Length**: 256 tokens
- **Embedding Dim**: 288
- **Attention Heads**: 6
- **KV Heads**: 6 (Multi-Head Attention)
- **Transformer Layers**: 6
- **Tokenizer**: Llama 2 SentencePiece (vocab size 32,000)

## Training

### Pre-training
Pre-trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Pre-trained weights available at [0rn0/llama2-15m-tinystories](https://huggingface.co/0rn0/llama2-15m-tinystories).

### Supervised Fine-Tuning (SFT)
Fine-tuned for 3 epochs on [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced), a curated instruction-tuning dataset with balanced happy/sad endings (50-50 split, ~325K examples).

**SFT Hyperparameters:**
- Optimizer: AdamW
- Learning Rate: 5e-5
- Batch Size: 64
- Epochs: 3
- Weight Decay: 0.1
- Max Sequence Length: 256

**Training Results:**
- Best Validation Loss: **1.1275** (Epoch 3)
- Final Training Loss: 1.0855
- Training Time: ~6 hours (1x A100 GPU)

## Usage

This is a **custom PyTorch model** (not transformers-compatible). Requires source code from the GitHub repository.

```bash
git clone https://github.com/aryandeore/llama2-sft.git
cd llama2-sft
uv sync
```

### Instruction Format
```
Write a story about: {topic}
With: {ending} ending

### Story:
```

Where `{ending}` is either `happy` or `sad`.

### Load Model
```python
from models.llama2 import Transformer

model = Transformer.from_pretrained("0rn0/llama2-15m-tinystories-sft")
model.eval()
```

### Generate Story
```bash
uv run python generate.py \
  --topic "a brave knight" \
  --ending sad \
  --temperature 0.7 \
  --max-tokens 192
```

## Examples

### Happy ending
**Prompt:** `Write a story about: a little girl and her dog`

> Once upon a time, there was a little girl named Sophie who had a small dog named Buddy. Every morning, Sophie and Buddy would run to the park together. One sunny day, Sophie accidentally dropped her favorite toy near the big oak tree. Buddy immediately began to sniff around the grass, searching everywhere. Finally, Buddy found it! Sophie was so happy and hugged her dog tightly. From that day on, Sophie knew that Buddy would always be there to help her find anything she lost. Sophie and Buddy played together in the park every single day, and they were the very best of friends forever.

### Sad ending
**Prompt:** `Write a story about: a boy who lost his favorite toy`

> Once there was a young boy named Thomas who loved his wooden toy train. He played with it every single day and took it everywhere he went. One afternoon, Thomas left his beloved toy train at the park by accident. When he realized it was missing, he rushed back to search for it. He looked everywhere but could not find his precious toy. Thomas felt very sad and disappointed. His parents tried to comfort him, but nothing seemed to help. Thomas went to bed that night with tears in his eyes, wishing he could have his toy train back.

## Limitations
- Generates short stories only (~150-200 words)
- Trained on synthetic data from TinyStories
- Limited vocabulary and narrative complexity
- Binary ending classification (happy/sad only)
- Small context window (256 tokens)

## Links

- **Pre-trained model:** [0rn0/llama2-15m-tinystories](https://huggingface.co/0rn0/llama2-15m-tinystories)
- **Dataset:** [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced)
- **GitHub:** [AryanDeore/llama2-sft](https://github.com/AryanDeore/llama2-sft)
- **Collection:** [Tiny Stories](https://huggingface.co/collections/0rn0/tiny-stories)
- **Related GPT-2 Models:** [0rn0/gpt2-30m-tinystories-sft](https://huggingface.co/0rn0/gpt2-30m-tinystories-sft), [0rn0/gpt2-125m-tinystories-sft](https://huggingface.co/0rn0/gpt2-125m-tinystories-sft)
