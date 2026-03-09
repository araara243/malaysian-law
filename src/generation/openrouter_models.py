"""
OpenRouter Model Fetching Utility

This module provides functions to fetch and filter free models
from the OpenRouter API for use in the MyLaw-RAG application.
"""

import logging
import os
from typing import List, Dict, Optional
import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


# Default option for auto-routing
DEFAULT_MODEL_OPTION = {
    "id": "openrouter/free",
    "name": "Auto-route to Best Free Model"
}


def fetch_free_models() -> List[Dict[str, str]]:
    """
    Fetch and filter free models from OpenRouter API.

    Returns:
        List of dictionaries with 'id' and 'name' keys for free models.
        Returns default option only if API key is missing or API call fails.
    """
    # Check if API key is configured
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        logger.warning("OPENROUTER_API_KEY not configured. Using default model option.")
        return [DEFAULT_MODEL_OPTION]

    try:
        # Make API request to fetch models
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Malaysian Legal RAG"
            },
            timeout=10
        )

        if response.status_code != 200:
            logger.error(
                f"OpenRouter API returned status {response.status_code}: "
                f"{response.text[:200]}"
            )
            return [DEFAULT_MODEL_OPTION]

        data = response.json()
        if "data" not in data:
            logger.error("OpenRouter API response missing 'data' field")
            return [DEFAULT_MODEL_OPTION]

        # Filter for free models (both prompt and completion prices are 0)
        free_models = []
        for model in data["data"]:
            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))

            # Check if model is completely free
            if prompt_price == 0 and completion_price == 0:
                free_models.append({
                    "id": model["id"],
                    "name": model.get("name", model["id"])
                })

        # Sort by name for better UX
        free_models.sort(key=lambda x: x["name"])

        # Prepend default option at the beginning
        return [DEFAULT_MODEL_OPTION] + free_models

    except requests.exceptions.Timeout:
        logger.error("OpenRouter API request timed out")
        return [DEFAULT_MODEL_OPTION]
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API request failed: {e}")
        return [DEFAULT_MODEL_OPTION]
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing OpenRouter API response: {e}")
        return [DEFAULT_MODEL_OPTION]
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {e}")
        return [DEFAULT_MODEL_OPTION]


def get_model_display_name(model_id: str, model_options: List[Dict[str, str]]) -> str:
    """
    Get a human-readable display name for a model ID.

    Args:
        model_id: The model ID to look up.
        model_options: List of model dictionaries from fetch_free_models().

    Returns:
        The display name, or the model_id if not found.
    """
    for option in model_options:
        if option["id"] == model_id:
            return option["name"]
    return model_id


if __name__ == "__main__":
    # Test the model fetching
    import sys
    from pathlib import Path

    # Add src to path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))

    print("=" * 70)
    print("OpenRouter Free Models Test")
    print("=" * 70)

    models = fetch_free_models()

    print(f"\nFound {len(models)} model option(s):\n")
    for i, model in enumerate(models, 1):
        print(f"{i}. ID: {model['id']}")
        print(f"   Name: {model['name']}")
        print()
