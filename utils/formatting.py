"""Instruction formatting templates for fine-tuning."""


def format_instruction(summary: str, ending: str) -> str:
    """Format user input as an instruction prompt.

    Args:
        summary: The story topic/summary
        ending: The ending type ("happy" or "sad")

    Returns:
        Formatted instruction prompt (without the story part)
    """
    return f"""Write a story about: {summary}
With: {ending} ending

### Story:
"""


def format_full_example(summary: str, ending: str, story: str) -> str:
    """Format a complete training example.

    Args:
        summary: The story topic/summary
        ending: The ending type ("happy" or "sad")
        story: The full story text

    Returns:
        Complete formatted example
    """
    return f"""Write a story about: {summary}
With: {ending} ending

### Story:
{story}"""
