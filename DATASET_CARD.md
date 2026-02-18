---
language:
- en
pretty_name: "TinyStories Instruct - Balanced"
tags:
- story-generation
- instruction-tuning
- balanced-dataset
- text-generation
- creative-writing
license: openrail
task_categories:
- text-generation
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: response
    dtype: string
  - name: ending
    dtype: string
  splits:
  - name: train
    num_examples: 324984
  - name: validation
    num_examples: 3542
  download_size: "2.5GB"
  dataset_size: "2.5GB"
---

# Dataset Card for TinyStories Instruct - Balanced

## Dataset Summary

**TinyStories Instruct - Balanced** is a curated, instruction-tuning dataset derived from [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). It contains short story generation examples with balanced happy/sad endings (50-50 split), making it ideal for fine-tuning language models to follow instructions and generate contextually appropriate narratives.

The dataset was created to address the original TinyStoriesInstruct's imbalance (92% happy / 8% sad endings) by:
1. Keeping ALL sad-ending examples (~162,492)
2. Randomly sampling an equal number of happy-ending examples (~162,492)
3. Resulting in a perfectly balanced 50-50 training set

## Dataset Details

### Data Format

Each example contains three fields:

```python
{
  "instruction": "Write a story about: {topic}\nWith: {ending} ending\n\n### Story:\n",
  "response": "{story_text}",
  "ending": "happy" or "sad"
}
```

### Example

**Instruction:**
```
Write a story about: Tom the gray cat and his best friend Lily love to play together
With: happy ending

### Story:
```

**Response:**
```
One day, Tom and Lily were playing with a big screen. They liked to draw pictures on it. Tom drew a beautiful tree and Lily drew some flowers. They were very happy together...
```

### Dataset Split

| Split | Examples | Happy | Sad |
|-------|----------|-------|-----|
| train | 324,984 | 162,492 (50%) | 162,492 (50%) |
| validation | 3,542 | 1,771 (50%) | 1,771 (50%) |
| **Total** | **328,526** | **164,263 (50%)** | **164,263 (50%)** |

## Dataset Creation

### Preprocessing
1. Parse raw TinyStoriesInstruct dataset (21M+ rows with `<|endoftext|>` delimiters)
2. Extract three fields per example: `Summary`, `Features`, `Story`
3. Classify ending type: "sad" if `Features` contains "BadEnding", else "happy"
4. Format instruction using template: `"Write a story about: {summary}\nWith: {ending} ending\n\n### Story:\n"`

### Balancing Strategy
```python
# Original distribution
happy: 19.32M (92%)
sad: 1.68M (8%)

# Balanced distribution (this dataset)
happy: 162,492 (50%) - random sample
sad: 162,492 (50%) - all examples
```

## Dataset Statistics

- **Total Examples**: 328,526
- **Train/Val Split**: 98.9% / 1.1%
- **Ending Balance**: Perfect 50-50
- **Average Summary Length**: ~30 words
- **Average Story Length**: ~180 words
- **Languages**: English

## Licensing Information

This dataset inherits the license from the original TinyStories dataset. Please refer to the [original dataset](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) for detailed license information.

## Dataset Limitations

1. **Short Stories Only**: ~180 words average; not suitable for long-form narrative tasks
2. **Synthetic Data**: Generated stories designed for model training, not human-authored
3. **Ending Simplicity**: Binary ending classification (happy/sad) - no nuanced emotional gradations
4. **Limited Diversity**: Stories feature common tropes and simple vocabulary (designed for language model pretraining)
5. **Imbalanced Original Source**: This dataset's balance is artificial; real-world stories may not follow this distribution
6. **Grammatical Quirks**: Some stories may have awkward phrasing (artifacts of generation process)

## Citation

If you use this dataset, please cite both this dataset and the original:

```bibtex
@dataset{tinystories_instruct_balanced_2026,
  title={TinyStories Instruct - Balanced: A Curated Story Generation Dataset},
  author={Aryan D},
  year={2026},
  url={https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced}
}

@dataset{tinystories_instruct,
  title={TinyStories Instruct},
  author={Eldan, Ronen},
  url={https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct}
}
```

## Additional Resources

- **Original TinyStories**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Original TinyStories Instruct**: [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
- **Related Paper**: [TinyStories: How Small Can Language Models Be and Still Speak Coherently?](https://arxiv.org/abs/2305.07759)
