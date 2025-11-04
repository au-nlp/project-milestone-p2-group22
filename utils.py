import json
import os
import pathlib
import shutil
from collections import Counter
from typing import Any, List

import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

dataset_name = "google/smol"


def get_or_build_smoldoc(
    configs: List[str], save_path: str, overwrite: bool = False, verbose: bool = True
) -> DatasetDict:
    """
    Load a SmolDoc DatasetDict from disk if available; otherwise build from Hugging Face and save.

    Args:
        configs: List of SmolDoc configuration names.
        save_path: Directory where the DatasetDict will be stored.
        overwrite: If True, forces re-download even if existing data is found.
        verbose: Print status messages.

    Returns:
        DatasetDict
    """
    if os.path.exists(save_path) and not overwrite:
        if verbose:
            print(
                f"📂 Found existing SmolDoc DatasetDict at {save_path}, loading from disk..."
            )
        return load_dataset_dict(save_path, verbose=verbose)

    if verbose:
        if overwrite:
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
