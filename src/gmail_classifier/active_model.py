"""Resolve the user's frozen, private experimental model and its review gate."""

import hashlib
import json

from .private_storage import private_home, private_path


def active_configuration():
    config = json.loads((private_home() / "active-model.json").read_text(encoding="utf-8"))
    model, report = private_path(config["model"]), private_path(config["report"])
    if hashlib.sha256(model.read_bytes()).hexdigest() != config["model_sha256"]:
        raise ValueError("Active model changed since freezing")
    return {**config, "model": model, "report": report}


def require_review(result, config):
    if not config.get("review_only"):
        return result
    return {**result, "proposed_label": result["label"], "action": "review", "label": "IA/Revisar",
            "review_reason": "Independent human validation pending; this is an experimental hypothesis"}
