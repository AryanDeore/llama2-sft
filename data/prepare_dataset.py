"""Parse raw TinyStoriesInstruct dataset and curate for fine-tuning.

This script:
1. Loads TinyStoriesInstruct from HuggingFace
2. Parses examples from raw text format
3. Extracts Summary, Features (ending type), and Story
4. Creates a curated dataset with explicit schema
5. Uploads to HuggingFace Hub
"""

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from typing import Generator


def parse_ending_type(features_str: str) -> str:
    """Extract ending type from Features field.

    Args:
        features_str: The Features field content

    Returns:
        "sad" if "BadEnding" in features, else "happy"
    """
    if features_str and "BadEnding" in features_str:
        return "sad"
    return "happy"


def parsed_examples_generator(split: str = "train"):
    """Stream raw TinyStoriesInstruct dataset and yield parsed examples.

    Accumulates rows until <|endoftext|>, extracts fields, yields dicts.

    Args:
        split: "train" or "test"

    Yields:
        Dict with keys: instruction, response, ending
    """
    raw_ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True)

    current_example = []

    for row in raw_ds:
        text = row["text"]
        current_example.append(text)

        if "<|endoftext|>" in text:
            # End of example - parse and yield
            full_text = "\n".join(current_example)

            # Extract fields
            summary = None
            features = None
            story = None

            lines = full_text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Features:"):
                    features = line.replace("Features:", "").strip()
                elif line.startswith("Story:"):
                    # Story spans rest of lines until endoftext
                    story_lines = lines[i+1:]
                    story = "\n".join(story_lines).replace("<|endoftext|>", "").strip()

            # Yield if complete
            if summary and story:
                ending = parse_ending_type(features)
                instruction = f"Write a story about: {summary}\nWith: {ending} ending\n\n### Story:\n"

                yield {
                    "instruction": instruction,
                    "response": story,
                    "ending": ending,
                }

            current_example = []


def create_balanced_dataset():
    """Create a balanced 50-50 happy/sad dataset and upload to HuggingFace Hub.

    Strategy:
    1. Parse all examples from the train split
    2. Separate into happy and sad lists
    3. Keep ALL sad examples
    4. Randomly sample the same number of happy examples
    5. Shuffle and upload
    """
    import random

    random.seed(42)

    features = Features({
        'instruction': Value('string'),
        'response': Value('string'),
        'ending': Value('string'),
    })

    # Step 1: Parse all train examples into happy/sad buckets
    print("Parsing train split (this will take a while for 21M rows)...")
    happy_examples = []
    sad_examples = []

    count = 0
    for ex in parsed_examples_generator(split="train"):
        if ex["ending"] == "sad":
            sad_examples.append(ex)
        else:
            happy_examples.append(ex)

        count += 1
        if count % 100_000 == 0:
            print(f"  Parsed {count:,} examples | happy: {len(happy_examples):,} | sad: {len(sad_examples):,}")

    print("\nTrain parsing complete:")
    print(f"  Total: {count:,}")
    print(f"  Happy: {len(happy_examples):,}")
    print(f"  Sad: {len(sad_examples):,}")

    # Step 2: Balance - keep all sad, sample equal number of happy
    num_sad = len(sad_examples)
    happy_sampled = random.sample(happy_examples, num_sad)

    print("\nBalanced training set:")
    print(f"  Sad: {num_sad:,}")
    print(f"  Happy (sampled): {len(happy_sampled):,}")
    print(f"  Total: {num_sad + len(happy_sampled):,}")

    # Step 3: Combine and shuffle
    all_train = sad_examples + happy_sampled
    random.shuffle(all_train)

    train_ds = Dataset.from_dict({
        "instruction": [ex["instruction"] for ex in all_train],
        "response": [ex["response"] for ex in all_train],
        "ending": [ex["ending"] for ex in all_train],
    }, features=features)

    # Step 4: Parse and balance validation split the same way
    print("\nParsing validation split...")
    happy_val = []
    sad_val = []

    for ex in parsed_examples_generator(split="validation"):
        if ex["ending"] == "sad":
            sad_val.append(ex)
        else:
            happy_val.append(ex)

    num_sad_val = len(sad_val)
    happy_val_sampled = random.sample(happy_val, min(num_sad_val, len(happy_val)))

    print(f"  Happy: {len(happy_val):,} | Sad: {num_sad_val:,}")
    print(f"  Balanced val: {num_sad_val + len(happy_val_sampled):,}")

    all_val = sad_val + happy_val_sampled
    random.shuffle(all_val)

    val_ds = Dataset.from_dict({
        "instruction": [ex["instruction"] for ex in all_val],
        "response": [ex["response"] for ex in all_val],
        "ending": [ex["ending"] for ex in all_val],
    }, features=features)

    # Step 5: Print samples
    print("\nSample examples:")
    for i in range(min(3, len(train_ds))):
        ex = train_ds[i]
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {ex['instruction'][:100]}...")
        print(f"Ending: {ex['ending']}")
        print(f"Response: {ex['response'][:100]}...")

    # Step 6: Push to hub
    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    print("\nPushing to hub...")
    print(f"  Train: {len(train_ds):,}")
    print(f"  Validation: {len(val_ds):,}")

    ds.push_to_hub("0rn0/tinystories-instruct-balanced", max_shard_size="500MB")
    print("Done!")


def test_parser_sample(split: str = "train", num_examples: int = 5):
    """Test parser on a small sample to verify logic."""
    print(f"Testing parser on {num_examples} examples from {split}...")
    count = 0
    for ex in parsed_examples_generator(split=split):
        count += 1
        print(f"\n--- Example {count} ---")
        print(f"Instruction: {ex['instruction'][:80]}...")
        print(f"Ending: {ex['ending']}")
        print(f"Response: {ex['response'][:80]}...")
        if count >= num_examples:
            break
    print(f"\n✓ Successfully parsed {count} examples")


def analyze_features(split: str = "train", num_examples: int = 100):
    """Analyze actual Features field to understand ending classifications."""
    from collections import Counter

    print(f"Analyzing Features field in {num_examples} examples from {split}...\n")

    feature_tags = Counter()
    ending_distribution = Counter()
    raw_ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True)

    current_example = []
    count = 0

    for row in raw_ds:
        text = row["text"]
        current_example.append(text)

        if "<|endoftext|>" in text:
            full_text = "\n".join(current_example)

            # Extract Features field
            features = None
            lines = full_text.split("\n")
            for line in lines:
                if line.startswith("Features:"):
                    features = line.replace("Features:", "").strip()
                    break

            count += 1

            # Count feature tags
            if features:
                # Parse individual tags
                tags = [tag.strip() for tag in features.split(",")]
                for tag in tags:
                    feature_tags[tag] += 1

                # Check our ending classification
                ending = "sad" if "BadEnding" in features else "happy"
            else:
                ending = "happy"  # default if no features

            ending_distribution[ending] += 1

            # Print sample
            if count <= 10:
                print(f"Example {count}:")
                print(f"  Features: {features}")
                print(f"  → Classified as: {ending}")

            current_example = []

            if count >= num_examples:
                break

    print("\n\n=== FEATURE TAG FREQUENCY ===")
    for tag, freq in feature_tags.most_common():
        pct = (freq / count) * 100
        print(f"  {tag}: {freq} ({pct:.1f}%)")

    print("\n=== ENDING DISTRIBUTION ===")
    for ending, freq in ending_distribution.most_common():
        pct = (freq / count) * 100
        print(f"  {ending}: {freq} ({pct:.1f}%)")

    print(f"\n✓ Analyzed {count} examples")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_parser_sample(num_examples=5)
    elif len(sys.argv) > 1 and sys.argv[1] == "analyze":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        analyze_features(num_examples=num)
    else:
        create_balanced_dataset()
