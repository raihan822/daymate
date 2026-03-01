"""
Without an __init__.py, your imports look like this:
from ai_integration.llm_model import load_llm
from ai_integration.llm_config import prompt_for_model

With a well-configured __init__.py, you can make them much shorter:
from ai_integration import load_llm
"""

# 1. Importing the main tools from internal files
from .llm_config import prompt_for_model, load_llm_objects
# 2. Define what is accessible when someone imports * # (This is optional but considered best practice)
__all__ = [
    'prompt_for_model', 'load_llm_objects'
]