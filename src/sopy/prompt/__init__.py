from typing import Optional


class Prompt:
    """
    A class to represent a prompt for the LLM.
    """

    def __init__(self, prompt: str):
        """
        Initializes the Prompt with the given string.

        Args:
            prompt (str): The prompt string.
        """
        self.prompt = prompt

    def __str__(self) -> str:
        """
        Returns the string representation of the Prompt.

        Returns:
            str: The prompt string.
        """
        return self.prompt

def cond(value: bool, descripton: Optional[str] = None) -> bool:
    return value


# TODO: convertion to Strands prompt / LangChain prompt, etc.
def make_prompt(prompt: str) -> Prompt:
    """
    Factory function to create a Prompt instance.

    Args:
        prompt (str): The prompt string.

    Returns:
        Prompt: An instance of the Prompt class.
    """
    return Prompt(prompt)

__all__ = [
    "Prompt",
    "make_prompt",
]