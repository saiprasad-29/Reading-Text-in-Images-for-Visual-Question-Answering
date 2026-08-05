"""TextVQA dataset loader (TextVQA 0.5.1 raw-json format).

Each item exposes:
    question_id : int
    image_id    : str
    question    : str
    image_path  : str
    answers     : list[str]   (10 human answers; empty for test split)
"""
import json
import os
from functools import lru_cache

from PIL import Image

from .config import load_config


def _resolve_image_path(image_dir: str, image_id: str) -> str:
    # TextVQA image_id maps to <image_id>.jpg in the open-images folder.
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(image_dir, image_id + ext)
        if os.path.exists(p):
            return p
    # fall back to the bare id (caller handles missing files)
    return os.path.join(image_dir, image_id + ".jpg")


class TextVQADataset:
    def __init__(self, split: str = "val", cfg: dict | None = None, limit: int = 0):
        self.cfg = cfg or load_config()
        self.split = split
        ann_path = self.cfg["paths"]["ann"][split]
        self.image_dir = self.cfg["paths"]["images"][split]
        with open(ann_path) as f:
            blob = json.load(f)
        records = blob["data"] if isinstance(blob, dict) and "data" in blob else blob

        # Keep only records whose image actually exists on disk, THEN apply limit --
        # otherwise limit can select records with no image file and every
        # downstream stage crashes.
        kept = []
        for r in records:
            p = _resolve_image_path(self.image_dir, r["image_id"])
            if os.path.exists(p):
                kept.append(r)
                if limit and len(kept) >= limit:
                    break
        self.records = kept

    def __len__(self):
        return len(self.records)

    def meta(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "question": r["question"],
            "image_path": _resolve_image_path(self.image_dir, r["image_id"]),
            "answers": r.get("answers", []),
        }

    def __getitem__(self, idx: int) -> dict:
        m = self.meta(idx)
        m["image"] = load_image(m["image_path"])
        return m


@lru_cache(maxsize=256)
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
