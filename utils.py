import os
import json
from typing import Any
import requests


def get_smoldoc_factuality(data_dir: str = "data") -> dict[str, Any]:
    """
    Download (if needed) and load the SmolDoc factuality ratings dataset.

    This function checks if the file `smoldoc-factuality-ratings.json`
    exists in the specified directory (default: "data"). If not, it
    downloads it from Hugging Face (`google/smol`) and saves it locally.

    Args:
        data_dir (str, optional): Directory where the file should be cached.
            Defaults to "data".

    Returns:
        dict: Parsed JSON content from `smoldoc-factuality-ratings.json`.

    Raises:
        requests.exceptions.RequestException: If the download fails.
        json.JSONDecodeError: If the file cannot be parsed as JSON.

    Example:
        >>> from utils import get_smoldoc_factuality
        >>> data = get_smoldoc_factuality()
        >>> import pandas as pd
        >>> df = pd.DataFrame(data)
        >>> df.head()
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "smoldoc-factuality-ratings.json")
    url = "https://huggingface.co/datasets/google/smol/resolve/main/smoldoc-factuality-ratings.json"

    if not os.path.exists(file_path):
        print(f"Downloading SmolDoc factuality ratings to {file_path} ...")
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Download complete.")
    else:
        print(f"Using cached file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data
