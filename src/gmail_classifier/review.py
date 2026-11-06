"""Convert manually reviewed, private collections to the training contract."""

import json
from pathlib import Path

from .dataset import FIELDS, validate_records
from .private_storage import private_path, write_private_json


def reviewed_records(collection: dict) -> list[dict]:
    if (not isinstance(collection, dict) or collection.get("format_version") != 1
            or collection.get("source") not in {"gmail_reviewed", "email_reviewed"}
            or not isinstance(collection.get("records"), list)):
        raise ValueError("Invalid review collection")
    rows = []
    for row in collection["records"]:
        if not isinstance(row, dict) or set(row) != FIELDS | {"human_reviewed", "privacy_reviewed", "exclude"}:
            raise ValueError("Unexpected review fields")
        if any(type(row[key]) is not bool for key in ("exclude", "human_reviewed", "privacy_reviewed")):
            raise ValueError("Review flags must be booleans")
        if row["exclude"]:
            continue
        if row["human_reviewed"] is not True or row["privacy_reviewed"] is not True:
            raise ValueError("Every included message requires human labeling and privacy review")
        if row["source"] != collection["source"]:
            raise ValueError("Unexpected source")
        rows.append({key: row[key] for key in FIELDS})
    return validate_records(rows, allow_private=True)


def prepare_dataset(collection_path: Path, output_path: Path) -> list[dict]:
    collection_path, output_path = private_path(collection_path), private_path(output_path)
    if collection_path == output_path:
        raise ValueError("Output must not overwrite the review collection")
    rows = reviewed_records(json.loads(collection_path.read_text(encoding="utf-8")))
    write_private_json(output_path, rows)
    return rows


def simulate(collection_path: Path, model_path: Path, output_path: Path) -> list[dict]:
    from .baseline import load_model
    collection_path, output_path = private_path(collection_path), private_path(output_path)
    if output_path in {collection_path, Path(model_path).resolve()} or output_path.exists():
        raise ValueError("Choose a new simulation output path")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if (not isinstance(collection, dict) or collection.get("format_version") != 1
            or collection.get("source") not in {"gmail_reviewed", "email_reviewed"}
            or not isinstance(collection.get("records"), list)):
        raise ValueError("Invalid review collection")
    model = load_model(model_path)
    predictions = []
    for row in collection["records"]:
        if set(row) != FIELDS | {"human_reviewed", "privacy_reviewed", "exclude"}:
            raise ValueError("Unexpected review fields")
        if type(row["exclude"]) is not bool or not isinstance(row["id"], str):
            raise ValueError("Invalid review record")
        if row["exclude"]:
            continue
        predictions.append({"id": row["id"], **model.predict(row["subject"], row["body"])})
    write_private_json(output_path, {"mailbox_changes": 0, "predictions": predictions})
    return predictions
