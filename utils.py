import json
import os
import pathlib
import shutil
from collections import Counter
from typing import Any, List

import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

dataset_name = "google/smol"

def get_extended_datasets(save_path: str = "data/smoldoc_datasets" , overwrite: bool = False) -> DatasetDict:
    """
    Builds all SmolDoc datasets and extends them with annotation notes per annotator
    :param save_path: Directory where the DatasetDict will be stored.
    :param overwrite: If True, forces re-download even if existing data is found.
    :return: The SmolDoc datasets with annotations notes per annotator.
    """
    datasets_dict = get_smoldoc_dataset(
        configs=list_smoldoc_configs(),
        save_path=save_path,
        force_download=overwrite
    )

    return append_factuality_notes(datasets_dict)


def append_factuality_notes(datasets_dict: DatasetDict) -> DatasetDict:
    """
     Extends the SmolDoc datasets with annotation notes per annotator

    :param datasets_dict: All SmolDoc datasets stored in a Dictionary.
    :return: The SmolDoc datasets with annotations notes per annotator.
    """
    json_ratings = get_smoldoc_factuality()
    factuality_ds = load_factuality_ratings(json_data=json_ratings)

    keep_cols = [
        'annotator_1_label', 'annotator_1_notes',
        'annotator_2_label', 'annotator_2_notes',
        'annotator_3_label', 'annotator_3_notes'
    ]

    # Build lookup dict: id -> annotations
    annot_fields_by_id = {
        k: {col: v for col, v in zip(keep_cols, vals)}
        for k, *vals in zip(
            factuality_ds["id"],
            *[factuality_ds[col] for col in keep_cols]
        )
    }

    def _join_annotations(batch):
        ids = batch["id"]
        # Vectorized per-column build
        return {
            col: [annot_fields_by_id[i][col] if i in annot_fields_by_id else None for i in ids]
            for col in keep_cols
        }

    # Join all configs
    return datasets_dict.map(_join_annotations, batched=True)


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


def get_smoldoc_dataset(
    configs: List[str], save_path: str, force_download: bool = False, verbose: bool = True
) -> DatasetDict:
    """
    Load a SmolDoc DatasetDict from disk if available; otherwise build from Hugging Face and save.

    Args:
        configs: List of SmolDoc configuration names.
        save_path: Directory where the DatasetDict will be stored.
        force_download: If True, forces re-download even if existing data is found.
        verbose: Print status messages.

    Returns:
        DatasetDict
    """
    if os.path.exists(save_path) and not force_download:
        if verbose:
            print(
                f"📂 Found existing SmolDoc DatasetDict at {save_path}, loading from disk..."
            )
        return load_dataset_dict(save_path, verbose=verbose)

    if verbose:
        if force_download:
            print(f"⚠️ Overwriting existing data at {save_path}...")
        print(f"🚀 Building SmolDoc DatasetDict from Hugging Face...")

    ds_dict = load_smoldoc_configs(dataset_name, configs, verbose=verbose)
    save_dataset_dict(ds_dict, save_path, overwrite=True)
    return ds_dict


def list_smoldoc_configs():
    url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    configs = sorted({item["config"] for item in data["splits"]})
    return [c for c in configs if c.lower().startswith("smoldoc__")]


LABEL_RANK = {
    "No Issues": 0,
    "N/A - Not Applicable": 0,  # treat N/A neutral (not an issue)
    "Not Sure": 1,
    "Minor Issue(s)": 2,
    "Clear Issue(s)": 3,
}


JSON_PATH = "smoldoc-factuality-ratings.json"


def load_factuality_ratings(json_data=None, json_path=JSON_PATH):
    if json_data is None:
        json_data = json.loads(pathlib.Path(json_path).read_text())
    rows = []
    for id, ann_dict in json_data.items():
        # Normalize annotator entries
        factuality_ratings = {}
        for k in ("1", "2", "3"):
            if (
                k in ann_dict
                and isinstance(ann_dict[k], list)
                and len(ann_dict[k]) >= 1
            ):
                label = ann_dict[k][0]
                notes = ann_dict[k][1] if len(ann_dict[k]) > 1 else ""
            else:
                label, notes = None, ""
            factuality_ratings[k] = (label, notes)

        labels = [
            factuality_ratings[annotator][0]
            for annotator in ("1", "2", "3")
            if factuality_ratings[annotator][0]
        ]
        # Majority label (fall back to first non-null)
        majority_label = Counter(labels).most_common(1)[0][0] if labels else None
        # Worst label by severity (max rank)
        worst_label = (
            max(labels, key=lambda L: LABEL_RANK.get(L, 0)) if labels else None
        )

        rows.append(
            {
                "id": id,
                "annotator_1_label": factuality_ratings["1"][0],
                "annotator_1_notes": factuality_ratings["1"][1],
                "annotator_2_label": factuality_ratings["2"][0],
                "annotator_2_notes": factuality_ratings["2"][1],
                "annotator_3_label": factuality_ratings["3"][0],
                "annotator_3_notes": factuality_ratings["3"][1],
                "majority_label": majority_label,
                "worst_label": worst_label,
                "worst_label_rank": LABEL_RANK.get(worst_label, 0),
            }
        )
    return Dataset.from_list(rows)


def load_smoldoc_configs(
    dataset_name: str, configs: List[str], verbose: bool = True
) -> DatasetDict:
    """Load multiple SmolDoc configurations from Hugging Face into a DatasetDict."""
    flat: dict[str, Dataset] = {}
    for cfg in configs:
        if verbose:
            print(f"📥 Loading config: {cfg} (split=train)")
        ds: Dataset = load_dataset(dataset_name, cfg, split="train")
        flat[cfg] = ds
    merged = DatasetDict(flat)  # keys are configs; values are Dataset (not DatasetDict)
    if verbose:
        print(
            f"✅ Loaded {len(merged)} configs into a flattened DatasetDict (values are Dataset)."
        )
    return merged


def save_dataset_dict(ds_dict: DatasetDict, path: str, overwrite: bool = False) -> None:
    """Save a DatasetDict to disk safely, with overwrite protection."""
    if not isinstance(ds_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(ds_dict)}")
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Path '{path}' already exists. Use overwrite=True to replace it."
            )
        shutil.rmtree(path)
    ds_dict.save_to_disk(path)
    print(f"💾 Saved DatasetDict with {len(ds_dict)} configs → {path}")


def load_dataset_dict(path: str, verbose: bool = True) -> DatasetDict:
    """Load a DatasetDict from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    ds = load_from_disk(path)
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {path}, got {type(ds)}")
    if verbose:
        print(f"📂 Loaded DatasetDict from {path} with {len(ds)} configs.")
    return ds

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


def print_iteration(id: str, question: str, ground_truth_answer: str, expected_answer: str, model_answer: str,
                    reasoning: str = None):
    print(f"Document ID: {id}")
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {ground_truth_answer}")
    print(f"Factually Incorrect Answer: {expected_answer}")
    print(model_answer)
    if reasoning:
        print("\n*** REASONING TRACE ***")
        print(f"{reasoning}\n")
    print(f"\n{'-' * 80}\n")