import requests
import json
import pathlib
import re
from collections import Counter
from datasets import Dataset

def list_smoldoc_configs(dataset="google/smol"):
    url = f"https://datasets-server.huggingface.co/splits?dataset={dataset}"
    r = requests.get(url); r.raise_for_status()
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

def load_topic_ratings(json_path=JSON_PATH):
    data = json.loads(pathlib.Path(json_path).read_text())
    rows = []
    for topic_key, ann_dict in data.items():
        # Extract a numeric id if present (e.g., "topic_183__xyz" -> 183)
        m = re.match(r"topic_(\d+)", topic_key)
        topic_id = int(m.group(1)) if m else None

        # Normalize annotator entries
        trip = {}
        for k in ("1", "2", "3"):
            if k in ann_dict and isinstance(ann_dict[k], list) and len(ann_dict[k]) >= 1:
                label = ann_dict[k][0]
                notes = ann_dict[k][1] if len(ann_dict[k]) > 1 else ""
            else:
                label, notes = None, ""
            trip[k] = (label, notes)

        labels = [trip[k][0] for k in ("1","2","3") if trip[k][0]]
        # Majority label (fall back to first non-null)
        majority_label = Counter(labels).most_common(1)[0][0] if labels else None
        # Worst label by severity (max rank)
        worst_label = max(labels, key=lambda L: LABEL_RANK.get(L, 0)) if labels else None

        rows.append({
            "topic_key": topic_key,
            "topic_id": topic_id,
            "annotator_1_label": trip["1"][0],
            "annotator_1_notes": trip["1"][1],
            "annotator_2_label": trip["2"][0],
            "annotator_2_notes": trip["2"][1],
            "annotator_3_label": trip["3"][0],
            "annotator_3_notes": trip["3"][1],
            "majority_label": majority_label,
            "worst_label": worst_label,
            "worst_label_rank": LABEL_RANK.get(worst_label, 0),
        })
    return Dataset.from_list(rows)