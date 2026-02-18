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
  - text: "Write a story about: a brave knight\nWith: happy ending\n\n### Story:"
    example_title: "Happy ending - Knight"
    outputs:
      - text: "Write a story about: a brave knight\nWith: happy ending\n\n### Story:\nOnce upon a time, there was a brave knight named Sir Arthur. He traveled across the kingdom helping people in need. One day, he saved a princess from a dragon's tower. The princess was so grateful and the two became best friends. They returned to the castle where the king rewarded Sir Arthur with gold and honor. Sir Arthur lived happily ever after, knowing he had made a difference in the world."
  - text: "Write a story about: a lost puppy\nWith: sad ending\n\n### Story:"
    example_title: "Sad ending - Puppy"
    outputs:
      - text: "Write a story about: a lost puppy\nWith: sad ending\n\n### Story:\nOnce there was a little puppy named Max who got separated from his family. He wandered the streets looking for them, but nobody could help him find his way home. Days passed and Max grew weaker and sadder. He missed his family so much and wished he could see them again. The puppy lay down under a tree, feeling very lonely and sad."
  - text: "Write a story about: a girl discovering magic\nWith: happy ending\n\n### Story:"
    example_title: "Happy ending - Magic"
    outputs:
      - text: "Write a story about: a girl discovering magic\nWith: happy ending\n\n### Story:\nA young girl named Emma found an old book in her grandmother's attic. When she opened it, magical words glowed on the pages. Emma spoke the words and suddenly flowers bloomed around her! She realized she had magical powers. With her newfound abilities, she helped her entire village and became known as the kindest person in the land. Everyone loved her and she was very happy."

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
