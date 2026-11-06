"""Training, held-out evaluation and trusted local artifact persistence."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from . import __version__
from .dataset import CATEGORIES, SPLITS, normalize_text, validate_records
from .policy import Policy


def pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])


def texts(rows):
    return [normalize_text(r["subject"], r["body"]) for r in rows]


@dataclass
class Baseline:
    spam: Pipeline
    category: Pipeline
    policy: Policy
    provenance: dict | None = None

    def predict(self, subject: str, body: str) -> dict:
        if not isinstance(subject, str) or not isinstance(body, str):
            raise ValueError("Subject and body must be strings")
        text = normalize_text(subject, body)
        if not text:
            raise ValueError("Message must contain text")
        spam_index = list(self.spam.classes_).index(True)
        probability = float(self.spam.predict_proba([text])[0, spam_index])
        known = self.spam.named_steps["tfidf"].transform([text]).nnz > 0
        if probability >= self.policy.spam_threshold and (self.provenance or {}).get("spam_suggestions_enabled") is False:
            return self.policy.decide(probability, known_text=False)
        if not known or probability > self.policy.legitimate_threshold:
            return self.policy.decide(probability, known_text=known)
        known_category = self.category.named_steps["tfidf"].transform([text]).nnz > 0
        if (self.provenance or {}).get("category_suggestions_enabled") is False:
            known_category = False
        probabilities = self.category.predict_proba([text])[0]
        index = int(probabilities.argmax())
        return self.policy.decide(probability, str(self.category.classes_[index]),
                                  float(probabilities[index]), known_text=known_category)


def train(records: list[dict], policy: Policy | None = None, *, allow_private=False,
          allow_ai=False, spam_reference: Baseline | None = None) -> Baseline:
    validate_records(records, allow_private=allow_private, allow_ai=allow_ai)
    rows = [r for r in records if r["split"] == "train"]
    legitimate = [r for r in rows if not r["is_spam"]]
    spam, category = pipeline(), pipeline()
    spam_source = records[0]["source"]
    if {r["is_spam"] for r in rows} == {False, True}:
        spam.fit(texts(rows), [r["is_spam"] for r in rows])
    elif allow_ai and records[0]["source"] == "gmail_ai_labeled" and spam_reference is not None:
        spam = deepcopy(spam_reference.spam)
        spam_source = (getattr(spam_reference, "provenance", None) or {}).get("spam_source", "reference_artifact")
    else:
        raise ValueError("Cannot train spam with one class; provide an explicit trusted reference")
    category.fit(texts(legitimate), [r["category"] for r in legitimate])
    provenance = {"spam_source": spam_source, "category_source": records[0]["source"],
                  "human_gold_labels": records[0]["source"] in {"gmail_reviewed", "email_reviewed"}}
    return Baseline(spam, category, policy or Policy(), provenance)


def metric_report(expected, predicted, labels):
    return {
        "labels": labels,
        "classification_report": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
    }


def evaluate(model: Baseline, records: list[dict], *, allow_private=False, allow_ai=False) -> dict:
    validate_records(records, allow_private=allow_private, allow_ai=allow_ai)
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    source = records[0]["source"]
    warning = ("Synthetic smoke test; not real-world quality" if source == "synthetic"
               else "Private reviewed sample; representativeness and thresholds require validation")
    if source == "gmail_ai_labeled":
        warning = "AI-labeled exploratory experiment; agreement with ChatGPT annotations, not human-verified accuracy"
    report = {"source": source, "dataset_sha256": fingerprint,
              "warning": warning, "model_provenance": getattr(model, "provenance", None),
              "version": __version__, "sklearn_version": sklearn.__version__,
              "seed": 42, "policy": asdict(model.policy), "splits": {}}
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        legitimate = [r for r in rows if not r["is_spam"]]
        index = list(model.spam.classes_).index(True)
        probabilities = model.spam.predict_proba(texts(rows))[:, index]
        predictions = [model.predict(r["subject"], r["body"]) for r in rows]
        accepted_spam = [p["label"] == "IA/SPAM" for p in predictions]
        accepted = sum(accepted_spam)
        category_predictions = model.category.predict(texts(legitimate)).tolist() if legitimate else []
        category_probabilities = model.category.predict_proba(texts(legitimate)) if legitimate else []
        category_labels = (sorted(set(r["category"] for r in legitimate) | set(category_predictions))
                           if source == "gmail_ai_labeled" else list(CATEGORIES))
        false_positive = sum(a and not r["is_spam"] for a, r in zip(accepted_spam, rows))
        report["splits"][split] = {
            "count": len(rows), "distribution": dict(Counter(
                "spam" if r["is_spam"] else r["category"] for r in rows)),
            "group_count": len({r["group_id"] for r in rows}),
            "spam_positive_count": sum(r["is_spam"] for r in rows),
            "spam_recall_assessable": any(r["is_spam"] for r in rows),
            "spam_at_0_5": metric_report([r["is_spam"] for r in rows],
                                         (probabilities >= 0.5).tolist(), [False, True]),
            "spam_at_policy": metric_report([r["is_spam"] for r in rows], accepted_spam, [False, True]),
            "spam_policy_precision": (accepted - false_positive) / accepted if accepted else None,
            "spam_policy_false_positives": false_positive,
            "category_on_true_legitimate": (metric_report(
                [r["category"] for r in legitimate], category_predictions,
                category_labels) if legitimate else None),
            "category_accuracy": (sum(r["category"] == prediction for r, prediction in
                                      zip(legitimate, category_predictions)) / len(legitimate)
                                  if legitimate else None),
            "review_rate": sum(p["action"] == "review" for p in predictions) / len(rows),
            "category_confidence_histogram": {
                "below_0_6": sum(bool(max(p) < 0.6) for p in category_probabilities),
                "at_least_0_6": sum(bool(max(p) >= 0.6) for p in category_probabilities),
            },
            "end_to_end_correct_label_rate": sum(
                p["label"] == ("IA/SPAM" if r["is_spam"] else f"IA/{r['category']}")
                for p, r in zip(predictions, rows)) / len(rows),
        }
    return report


def save_model(model: Baseline, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format_version": 1, "package_version": __version__,
                 "sklearn_version": sklearn.__version__, "model": model}, path)


def load_model(path: str | Path) -> Baseline:
    # joblib/pickle can execute code. Only load artifacts you trained and trust.
    artifact = joblib.load(path)
    if (not isinstance(artifact, dict) or artifact.get("format_version") != 1
            or artifact.get("package_version") != __version__
            or artifact.get("sklearn_version") != sklearn.__version__
            or not isinstance(artifact.get("model"), Baseline)):
        raise ValueError("Incompatible model artifact; retrain in this environment")
    return artifact["model"]
