"""Strict validation of synthetic and explicitly opted-in private datasets."""

import json
from pathlib import Path

CATEGORIES = (
    "Financeiro", "Suporte", "RH", "Sistemas", "Comercial",
    "Administrativo", "Notificações", "Pessoal", "Outros",
)
SPLITS = ("train", "validation", "test")
FIELDS = {"id", "group_id", "subject", "body", "is_spam", "category", "source", "split"}


def normalize_text(subject: str, body: str) -> str:
    return " ".join(f"{subject} {body}".casefold().split())


def validate_records(records: list[dict], *, allow_private=False, allow_ai=False) -> list[dict]:
    if allow_ai and not allow_private:
        raise ValueError("AI labels require private-data opt-in")
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must be a non-empty JSON array")
    ids, texts, groups = set(), set(), {}
    for index, row in enumerate(records):
        # Error messages deliberately omit email content.
        prefix = f"Record {index}"
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"{prefix}: unexpected or missing fields")
        for field in ("id", "group_id", "subject", "body", "source", "split"):
            if not isinstance(row[field], str):
                raise ValueError(f"{prefix}: {field} must be a string")
        if not row["id"].strip() or not row["group_id"].strip():
            raise ValueError(f"{prefix}: empty identifier")
        text = normalize_text(row["subject"], row["body"])
        if not text:
            raise ValueError(f"{prefix}: empty message")
        if type(row["is_spam"]) is not bool:
            raise ValueError(f"{prefix}: is_spam must be a boolean")
        sources = {"synthetic", "gmail_reviewed", "email_reviewed"} if allow_private else {"synthetic"}
        if allow_ai:
            sources.add("gmail_ai_labeled")
        if row["source"] not in sources or row["split"] not in SPLITS:
            raise ValueError(f"{prefix}: unsupported source or split (private data requires opt-in)")
        if row["is_spam"]:
            if row["category"] is not None:
                raise ValueError(f"{prefix}: spam category must be null")
        elif row["category"] not in CATEGORIES:
            raise ValueError(f"{prefix}: invalid legitimate category")
        if row["id"] in ids or text in texts:
            raise ValueError(f"{prefix}: duplicate id or normalized message")
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError(f"{prefix}: group crosses split boundary")
        ids.add(row["id"])
        texts.add(text)
        groups[row["group_id"]] = row["split"]
    if len({r["source"] for r in records}) != 1:
        raise ValueError("Do not mix synthetic and private data in one evaluation")
    if records[0]["source"] in {"gmail_ai_labeled", "email_reviewed"}:
        # Exploratory AI labels have incomplete class coverage; do not invent examples.
        training = [r for r in records if r["split"] == "train"]
        categories = {r["category"] for r in training if not r["is_spam"]}
        if len(categories) < 2:
            raise ValueError("AI experiment requires two legitimate categories in training")
        for split in SPLITS:
            rows = [r for r in records if r["split"] == split]
            if not rows:
                raise ValueError(f"{split}: requires held-out groups")
            if any(not r["is_spam"] and r["category"] not in categories for r in rows):
                raise ValueError(f"{split}: category missing from training")
        return records
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        if {r["is_spam"] for r in rows} != {True, False}:
            raise ValueError(f"{split}: requires spam and legitimate messages")
        if {r["category"] for r in rows if not r["is_spam"]} != set(CATEGORIES):
            raise ValueError(f"{split}: requires every legitimate category")
    return records


def load_dataset(path: str | Path, *, allow_private=False, allow_ai=False) -> list[dict]:
    if allow_private:
        from .private_storage import private_path
        path = private_path(path)
    return validate_records(json.loads(Path(path).read_text(encoding="utf-8")),
                            allow_private=allow_private, allow_ai=allow_ai)
