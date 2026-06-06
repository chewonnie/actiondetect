"""
Mapping from ETRI-Activity3D 55-class labels to 12 core activity classes.

Label list origin: YOWOv3.zip:YOWOv3/config/etri_idx2name_{ko,en}.yaml
Dataset: ETRI EPreTX (https://epretx.etri.re.kr/dataDetail?id=12)
NOTE: The YOWOv3 codebase/models are explicitly out of scope —
      only the label list was taken from it.

action_idx is 0-based and matches src/dataset.py:parse_action_index (A001 -> 0).
"""

import csv
from typing import Optional

# 12 core class names; index = core_idx
CORE_NAMES: list[str] = [
    "eating",               # 0
    "drinking",             # 1
    "medicine",             # 2
    "cooking_kitchen",      # 3
    "hygiene_grooming",     # 4
    "housework",            # 5
    "phone",                # 6
    "sedentary_screen",     # 7
    "exercise",             # 8
    "mobility",             # 9
    "posture_transition",   # 10
    "other_social",         # 11
]

# Confirmed mapping: ETRI action_idx (0-based, 0..54) -> core_idx (0..11)
# All 55 actions map to exactly one of the 12 core classes.
ETRI_TO_CORE: dict[int, int] = {
    # core 0 — eating
    0: 0,
    # core 1 — drinking
    3: 1,
    # core 2 — medicine
    2: 2,
    # core 3 — cooking_kitchen
    1: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3,
    # core 4 — hygiene_grooming
    9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
    15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4,
    # core 5 — housework
    21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5,
    # core 6 — phone
    34: 6, 35: 6,
    # core 7 — sedentary_screen
    30: 7, 31: 7, 32: 7, 33: 7, 36: 7,
    # core 8 — exercise
    40: 8, 41: 8, 42: 8,
    # core 9 — mobility
    29: 9, 51: 9, 52: 9,
    # core 10 — posture_transition
    53: 10, 54: 10,
    # core 11 — other_social
    37: 11, 38: 11, 39: 11, 43: 11, 44: 11,
    45: 11, 46: 11, 47: 11, 48: 11, 49: 11, 50: 11,
}


def remap(action_idx: int) -> Optional[int]:
    """Return core_idx for a valid ETRI action_idx (0..54), else None."""
    return ETRI_TO_CORE.get(action_idx)


def load_actions_csv(path: str = "pipeline/etri_actions.csv") -> list[dict]:
    """
    Load the ETRI actions CSV for cross-checking purposes.
    Returns a list of row dicts with keys: action_idx, file_token, name_en, name_ko.
    ETRI_TO_CORE (hard-coded above) is the source of truth, not this CSV.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        # Skip '#' provenance comment lines (see etri_actions.csv header).
        reader = csv.DictReader(ln for ln in f if not ln.lstrip().startswith("#"))
        for row in reader:
            row["action_idx"] = int(row["action_idx"])
            rows.append(row)
    return rows
