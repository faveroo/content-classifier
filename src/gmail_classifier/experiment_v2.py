"""Private, auditable progression: mixed real corpora, AI topics, validation, frozen test."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .baseline import Baseline, load_model, save_model
from .dataset import CATEGORIES
from .email_corpora import prepare_rows
from .mail_input import canonical_text, template_text
from .message_model import evaluate_probabilities, text_rows
from .policy import Policy
from .private_storage import private_home, private_path, write_private_json
from .text_privacy import redact
from .text_features import clean_text

SEED = 42
VARIANTS = ("word", "char", "hybrid")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, *, overwrite=False):
    return write_private_json(path, value, overwrite=overwrite)


def split_development(rows, *, thematic=False):
    """Split connected groups by source/class signature, never splitting a group."""
    rows = deepcopy(rows)
    groups, strata = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for group, members in groups.items():
        signature = tuple(sorted({(r.get("origin", "topic"), r["category"] if thematic else str(r["is_spam"])) for r in members}))
        strata[signature].append(group)
    rng = random.Random(SEED)
    for signature in sorted(strata):
        ids = sorted(strata[signature])
        rng.shuffle(ids)
        nv = max(1, round(len(ids) * (.25 if thematic else .15))) if len(ids) >= 2 else 0
        nd = max(1, round(len(ids) * .15)) if len(ids) >= 3 and not thematic else 0
        for i, group in enumerate(ids):
            split = "validation" if i < nv else "diagnostic" if i < nv + nd else "train"
            for row in groups[group]:
                row["split"] = split
    return rows


def prepare(directory, public_dataset):
    directory = private_path(directory)
    if directory.exists():
        raise ValueError("Choose a new experiment directory")
    collection = read(private_home() / "email-category-review.json")
    old = read(private_home() / "dataset-ai-v1.json")
    if len(collection["records"]) != 180 or any(r["source"] != "gmail_ai_labeled" for r in old):
        raise ValueError("Unexpected annotation inputs")
    public = read(public_dataset)
    private_keys = {canonical_text(r["subject"], r["body"]) for r in collection["records"]}
    source_keys = {canonical_text(redact(r["subject"]), redact(r["body"])) for r in public["records"]}
    if private_keys - source_keys:
        raise ValueError("Cannot trace every thematic example to public source groups; refusing possible cross-stage leakage")
    # Quarantine ALL topic sample groups from spam development, including sealed test.
    excluded = {r["group_id"] for r in public["records"]
                if canonical_text(redact(r["subject"]), redact(r["body"])) in private_keys}
    public_rows = [{**r, "origin": r["source"], "source": "real_development"}
                   for r in public["records"] if r["group_id"] not in excluded]
    grouped, stats = prepare_rows(public_rows, training_source="real_development", external_source=None)
    development = split_development(grouped)
    directory.mkdir(parents=True)
    save(directory / "queue.json", collection)
    save(directory / "old-ai-categories.json", old)
    save(directory / "development.json", {"source": "real_multi_source_development", "records": development,
        "source_dataset_sha256": digest(public_dataset), "excluded_topic_groups": len(excluded),
        "excluded_topic_rows": len(public["records"]) - len(public_rows), "preparation": stats})
    protocol = {"format_version": 1, "seed": SEED, "variants": list(VARIANTS),
        "authorization": "User requested following the roadmap with full project autonomy on 2026-11-05; AI annotations are exploratory, never human approvals.",
        "spam_threshold_grid": [.5, .6, .7, .8, .85, .9, .92, .94, .96, .98, .99, .995, .999, 1.0],
        "legitimate_threshold_grid": [.01, .02, .05, .1, .15, .2, .25, .3],
        "category_threshold_grid": [.4, .5, .6, .7, .8, .9, .95],
        "max_validation_spam_fpr_per_source": .01, "min_validation_spam_precision": .98,
        "min_validation_spam_suggestions": 10, "min_category_accepted": 10,
        "min_category_validation_agreement": .9,
        "selection": "Spam: maximize worst-source validation macro F1, then global macro F1. Topic: validation macro F1 over trained classes. Policy uses validation only.",
        "topic_test_indices": [i for i, r in enumerate(collection["records"]) if r["split"] == "test"],
        "topic_test_sha256": hashlib.sha256(json.dumps([r for r in collection["records"] if r["split"] == "test"], sort_keys=True).encode()).hexdigest(),
        "queue_sha256": digest(directory / "queue.json"), "development_sha256": digest(directory / "development.json"),
        "human_gold_labels": False, "recent_private_test": "Pending read-only collection/re-authentication; never substitute historical data silently."}
    save(directory / "protocol.json", protocol)
    save(directory / "annotations.json", {"source": "email_ai_labeled", "annotator": "ChatGPT", "human_reviewed": False,
        "privacy_reviewed": False, "context": "Subject and body excerpts; ambiguous content excluded or inspected further.", "labels": {}})
    return {"directory": str(directory), "spam_count": len(development),
            "splits": dict(Counter(r["split"] for r in development)), "topic_reserved": len(protocol["topic_test_indices"])}


def view(directory, start, count, *, test=False, indices=None):
    directory = private_path(directory)
    protocol, collection = read(directory / "protocol.json"), read(directory / "queue.json")
    if digest(directory / "queue.json") != protocol["queue_sha256"]:
        raise ValueError("Annotation queue changed after reservation")
    if test and not (directory / "freeze.json").exists():
        raise ValueError("Freeze selected models before reading the thematic test")
    if test:
        verify_freeze(directory)
    rows = [(i, r) for i, r in enumerate(collection["records"])
            if (i in protocol["topic_test_indices"]) == test]
    rows = [(i, r) for i, r in rows if i in indices] if indices else rows[start:start + count]
    return [{"i": i, "subject": r["subject"], "body": r["body"] if indices else r["body"][:1000],
             "tail": r["body"][-180:] if not indices and len(r["body"]) > 1000 else ""} for i, r in rows]


def annotate(directory, entries):
    directory = private_path(directory)
    protocol, annotation = read(directory / "protocol.json"), read(directory / "annotations.json")
    frozen = (directory / "freeze.json").exists()
    if frozen:
        verify_freeze(directory)
    for entry in entries:
        index = entry["index"]
        if type(index) is not int or not 0 <= index < 180:
            raise ValueError("Invalid annotation index")
        if (index in protocol["topic_test_indices"]) != frozen:
            raise ValueError("Development annotations precede freezing; final annotations follow it")
        if entry.get("category") not in {*CATEGORIES, None} or not 0 <= entry["confidence"] <= 1 or not entry.get("reason"):
            raise ValueError("Invalid category annotation/provenance")
        key = str(index)
        if key in annotation["labels"] and annotation["labels"][key] != entry:
            raise ValueError("Annotation exists; preserve the original manifest")
        annotation["labels"][key] = entry
    save(directory / f"annotations-backup-{len(annotation['labels'])}.json", annotation)
    save(directory / "annotations.json", annotation, overwrite=True)
    return {"annotated": len(annotation["labels"]), "human_reviewed": False}


def candidate(variant, *, thematic=False):
    common = dict(preprocessor=clean_text, lowercase=False, max_features=80000,
                  min_df=2, sublinear_tf=True, dtype=np.float32)
    word = TfidfVectorizer(ngram_range=(1, 2), **common)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
    feature = {"word": word, "char": char}.get(variant)
    if variant == "hybrid":
        feature = FeatureUnion([("word", word), ("char", char)])
    if feature is None:
        raise ValueError("Unknown classic model variant")
    return Pipeline([("tfidf", feature), ("classifier", LogisticRegression(
        C=1.0, max_iter=1500, class_weight="balanced", random_state=SEED,
        solver="lbfgs" if thematic else "liblinear"))])


def binary_metrics(model, rows, threshold=.5):
    probability = model.predict_proba(text_rows(rows))[:, list(model.classes_).index(True)]
    return evaluate_probabilities([r["is_spam"] for r in rows], probability, threshold)


def choose_spam_threshold(rows, probability, protocol):
    eligible = []
    truth = np.array([r["is_spam"] for r in rows])
    for threshold in protocol["spam_threshold_grid"]:
        accepted = probability >= threshold
        metrics = evaluate_probabilities(truth, probability, threshold)
        per_source = {}
        for source in sorted({r["origin"] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r["origin"] == source]
            per_source[source] = evaluate_probabilities(truth[indices], probability[indices], threshold)
        if (accepted.sum() >= protocol["min_validation_spam_suggestions"]
                and (metrics["spam_precision"] or 0) >= protocol["min_validation_spam_precision"]
                and all(m["false_positive_rate"] is not None and m["false_positive_rate"] <= protocol["max_validation_spam_fpr_per_source"] for m in per_source.values())):
            eligible.append((metrics["spam_recall"], threshold, metrics, per_source))
    if not eligible:
        return {"threshold": 1.0, "feasible": False, "reason": "No non-abstaining threshold met validation constraints"}
    _, threshold, metrics, per_source = max(eligible, key=lambda x: (x[0], x[1]))
    return {"threshold": threshold, "feasible": True, "validation": metrics, "by_source": per_source}


def fit_spam(directory):
    directory = private_path(directory)
    if (directory / "spam-selection.json").exists() or (directory / "freeze.json").exists():
        raise ValueError("Selection already exists; start a new experiment instead")
    protocol = read(directory / "protocol.json")
    if digest(directory / "development.json") != protocol["development_sha256"]:
        raise ValueError("Development dataset changed")
    rows = read(directory / "development.json")["records"]
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    results = {}
    for variant in VARIANTS:
        print(json.dumps({"stage": "spam_fit", "variant": variant}), flush=True)
        model = candidate(variant)
        model.fit(text_rows(train), [r["is_spam"] for r in train])
        probability = model.predict_proba(text_rows(validation))[:, list(model.classes_).index(True)]
        metrics = evaluate_probabilities([r["is_spam"] for r in validation], probability)
        by_source = {}
        for source in sorted({r["origin"] for r in validation}):
            indices = [i for i, r in enumerate(validation) if r["origin"] == source]
            by_source[source] = evaluate_probabilities([validation[i]["is_spam"] for i in indices], probability[indices])
        policy = choose_spam_threshold(validation, probability, protocol)
        results[variant] = {"validation": metrics, "by_source": by_source, "spam_policy": policy,
                            "worst_source_macro_f1": min(m["classification_report"]["macro avg"]["f1-score"] for m in by_source.values())}
        joblib.dump(model, directory / f"spam-{variant}.joblib")
        print(json.dumps({"stage": "spam_validation", "variant": variant,
                          "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
                          "threshold": policy["threshold"]}), flush=True)
    selected = max(VARIANTS, key=lambda v: (results[v]["worst_source_macro_f1"],
        results[v]["validation"]["classification_report"]["macro avg"]["f1-score"], -VARIANTS.index(v)))
    model = joblib.load(directory / f"spam-{selected}.joblib")
    probability = model.predict_proba(text_rows(validation))[:, list(model.classes_).index(True)]
    truth = np.array([r["is_spam"] for r in validation])
    legitimate_thresholds = []
    for threshold in protocol["legitimate_threshold_grid"]:
        mask = probability <= threshold
        count = int(mask.sum())
        wrong = int((mask & truth).sum())
        if count >= 20 and wrong / count <= .01 and wrong / max(1, truth.sum()) <= .01:
            legitimate_thresholds.append((count, threshold, wrong))
    legitimate = max(legitimate_thresholds)[1] if legitimate_thresholds else 0.0
    result = {"selected": selected, "candidates": results,
              "spam_threshold": results[selected]["spam_policy"]["threshold"],
              "legitimate_threshold": legitimate, "selection_basis": "validation_only",
              "selected_pipeline_sha256": digest(directory / f"spam-{selected}.joblib"),
              "protocol_sha256": digest(directory / "protocol.json")}
    save(directory / "spam-selection.json", result)
    return {"selected_spam": selected, "spam_threshold": result["spam_threshold"], "legitimate_threshold": legitimate}


def connect_topic_groups(rows):
    from .mail_input import content_group_keys
    rows = deepcopy(rows)
    parent, lookup = list(range(len(rows))), {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        keys = {(kind, value.casefold()) for kind, value in content_group_keys(row)}
        keys.add(("declared_group", row["group_id"]))
        for key in keys:
            if key in lookup:
                parent[root(i)] = root(lookup[key])
            else:
                lookup[key] = i
    for i, row in enumerate(rows):
        row["group_id"] = "topic-" + str(root(i))
    return rows


def topic_development(directory):
    protocol = read(directory / "protocol.json")
    annotations = read(directory / "annotations.json")["labels"]
    queue = read(directory / "queue.json")["records"]
    required = set(range(len(queue))) - set(protocol["topic_test_indices"])
    if not required.issubset({int(i) for i in annotations}):
        raise ValueError("Finish development annotations before training topics")
    rows = []
    for i, row in enumerate(queue):
        annotation = annotations.get(str(i), {})
        rows.append({**row, "origin": "public_enron_ai", "reserved": i in protocol["topic_test_indices"],
                     "category": annotation.get("category"), "annotation_index": i,
                     "annotation_confidence": annotation.get("confidence", 0), "source": "email_ai_labeled"})
    for row in read(directory / "old-ai-categories.json"):
        if not row["is_spam"]:
            rows.append({**row, "origin": "private_gmail_ai", "reserved": False,
                         "annotation_confidence": None, "include_prior_ai_label": True, "source": "email_ai_labeled"})
    rows = connect_topic_groups(rows)
    reserved_groups = {r["group_id"] for r in rows if r["reserved"]}
    dev = [r for r in rows if not r["reserved"] and r["group_id"] not in reserved_groups
           and r["category"] in CATEGORIES and (r.get("include_prior_ai_label") or r["annotation_confidence"] >= .65)]
    # Collapse exact duplicates after grouping; never count clones as independent labels.
    unique = {}
    for row in dev:
        key = canonical_text(row["subject"], row["body"])
        if key in unique and unique[key]["category"] != row["category"]:
            raise ValueError("Conflicting topic annotations for identical text")
        unique.setdefault(key, row)
    dev = split_development(list(unique.values()), thematic=True)
    trained = {r["category"] for r in dev if r["split"] == "train"}
    if len(trained) < 2 or not any(r["split"] == "validation" for r in dev):
        raise ValueError("Insufficient independent topic development groups")
    return dev, {"included": len(dev), "reserved": sum(r["reserved"] for r in rows),
                 "excluded_or_deduplicated": len(rows) - len(dev) - sum(r["reserved"] for r in rows),
                 "trained_categories": sorted(trained), "missing_training_categories": sorted(set(CATEGORIES) - trained)}


def fit_topics_and_freeze(directory, *, allow_private=False, allow_ai=False):
    if not allow_private or not allow_ai:
        raise ValueError("Real AI topic training requires --allow-private --allow-ai-labels")
    directory = private_path(directory)
    if (directory / "freeze.json").exists():
        raise ValueError("Model already frozen")
    protocol, spam_selection = read(directory / "protocol.json"), read(directory / "spam-selection.json")
    if digest(directory / "protocol.json") != spam_selection["protocol_sha256"]:
        raise ValueError("Selection protocol changed")
    dev, coverage = topic_development(directory)
    train = [r for r in dev if r["split"] == "train"]
    validation = [r for r in dev if r["split"] == "validation"]
    save(directory / "category-development.json", {"source": "email_ai_labeled", "records": dev, "coverage": coverage})
    results = {}
    for variant in VARIANTS:
        model = candidate(variant, thematic=True)
        model.fit(text_rows(train), [r["category"] for r in train])
        guessed = model.predict(text_rows(validation))
        truth = [r["category"] for r in validation]
        probabilities = model.predict_proba(text_rows(validation)).max(axis=1)
        report = classification_report(truth, guessed, labels=model.classes_.tolist(), output_dict=True, zero_division=0)
        feasible = []
        for threshold in protocol["category_threshold_grid"]:
            mask = probabilities >= threshold
            if mask.sum() >= protocol["min_category_accepted"]:
                accuracy = float(np.mean(guessed[mask] == np.asarray(truth)[mask]))
                if accuracy >= protocol["min_category_validation_agreement"]:
                    feasible.append((int(mask.sum()), threshold, accuracy))
        policy = max(feasible) if feasible else (0, 1.0, None)
        results[variant] = {"validation_count": len(validation), "classification_report": report,
            "macro_f1": report["macro avg"]["f1-score"], "category_threshold": policy[1],
            "validation_accepted": policy[0], "accepted_agreement": policy[2], "threshold_feasible": bool(feasible)}
        joblib.dump(model, directory / f"category-{variant}.joblib")
    selected = max(VARIANTS, key=lambda v: (results[v]["macro_f1"], -VARIANTS.index(v)))
    category = joblib.load(directory / f"category-{selected}.joblib")
    spam_path = directory / f"spam-{spam_selection['selected']}.joblib"
    if digest(spam_path) != spam_selection["selected_pipeline_sha256"]:
        raise ValueError("Selected spam pipeline changed")
    spam = joblib.load(spam_path)
    policy = Policy(spam_selection["spam_threshold"], spam_selection["legitimate_threshold"], results[selected]["category_threshold"])
    provenance = {"spam_source": "Enron-Spam + SpamAssassin real development",
        "category_source": "email_ai_labeled", "human_gold_labels": False, "human_validation_pending": True,
        "spam_features": spam_selection["selected"], "category_features": selected,
        "spam_suggestions_enabled": spam_selection["candidates"][spam_selection["selected"]]["spam_policy"]["feasible"],
        "category_suggestions_enabled": results[selected]["threshold_feasible"],
        "selection_basis": "validation_only"}
    model = Baseline(spam, category, policy, provenance)
    model_path = directory / "messages-real-v2.joblib"
    save_model(model, model_path)
    selection = {"selected": selected, "candidates": results, "coverage": coverage,
                 "source": "email_ai_labeled", "human_gold_labels": False,
                 "warning": "Agreement with AI annotations, not independently human-verified accuracy"}
    save(directory / "category-selection.json", selection)
    annotations = read(directory / "annotations.json")
    save(directory / "annotations-development-frozen.json", annotations)
    freeze = {"model_sha256": digest(model_path), "protocol_sha256": digest(directory / "protocol.json"),
        "category_development_sha256": digest(directory / "category-development.json"),
        "annotations_development_sha256": digest(directory / "annotations-development-frozen.json"),
        "queue_sha256": digest(directory / "queue.json"), "policy": asdict(policy), "provenance": provenance,
        "note": "Frozen before thematic test annotation or final simulation; do not change model/thresholds based on final outcomes."}
    save(directory / "freeze.json", freeze)
    return {"model": str(model_path), "selected_category": selected, "policy": asdict(policy), "coverage": coverage}


def verify_freeze(directory):
    directory = private_path(directory)
    freeze = read(directory / "freeze.json")
    checks = {"model_sha256": "messages-real-v2.joblib", "protocol_sha256": "protocol.json",
              "category_development_sha256": "category-development.json", "queue_sha256": "queue.json",
              "annotations_development_sha256": "annotations-development-frozen.json"}
    for key, filename in checks.items():
        if freeze.get(key) != digest(directory / filename):
            raise ValueError("A frozen model or input changed; final evaluation refused")
    return freeze


def topic_metrics(model, rows):
    if not rows:
        return None
    expected = [r["category"] for r in rows]
    guesses = model.category.predict(text_rows(rows))
    routed = [model.predict(r["subject"], r["body"]) for r in rows]
    return {"count": len(rows), "distribution": dict(Counter(expected)),
        "accuracy": float(np.mean(guesses == np.asarray(expected))),
        "classification_report": classification_report(expected, guesses, labels=list(CATEGORIES), output_dict=True, zero_division=0),
        "test_categories_without_training": sorted(set(expected) - set(model.category.classes_)),
        "categories_not_assessed": sorted(set(CATEGORIES) - set(expected)),
        "review_rate": sum(r["action"] == "review" for r in routed) / len(rows),
        "correct_accepted_label_rate": sum(r["label"] == f"IA/{e}" for r, e in zip(routed, expected)) / len(rows),
        "false_spam_suggestions_on_legitimate": sum(r["label"] == "IA/SPAM" for r in routed)}


def evaluate(directory):
    directory = private_path(directory)
    freeze = verify_freeze(directory)
    if (directory / "evaluation-started.json").exists():
        raise ValueError("Final evaluation already started; do not adapt to the held-out results")
    protocol = read(directory / "protocol.json")
    annotations = read(directory / "annotations.json")
    frozen_annotations = read(directory / "annotations-development-frozen.json")
    if any(annotations["labels"].get(k) != v for k, v in frozen_annotations["labels"].items()):
        raise ValueError("Development annotations changed after model selection")
    if not set(map(str, protocol["topic_test_indices"])).issubset(annotations["labels"]):
        raise ValueError("Complete the reserved thematic annotations after freezing")
    save(directory / "evaluation-started.json", {"model_sha256": freeze["model_sha256"], "one_time": True})
    model = load_model(directory / "messages-real-v2.joblib")
    queue = read(directory / "queue.json")["records"]
    final_rows, excluded = [], []
    for index in protocol["topic_test_indices"]:
        annotation = annotations["labels"][str(index)]
        if annotation["category"] is None or annotation["confidence"] < .65:
            excluded.append(index)
        else:
            final_rows.append({**queue[index], "category": annotation["category"], "source": "email_ai_labeled"})
    development = read(directory / "development.json")["records"]
    diagnostic = [r for r in development if r["split"] == "diagnostic"]
    report = {"source": "real_development_and_ai_topic_test", "human_gold_labels": False,
        "warning": "Thematic metrics measure AI-label agreement. Spam diagnostic reuses historical public sources. No independent human validation or recent-mail accuracy is claimed.",
        "model_sha256": freeze["model_sha256"], "policy": asdict(model.policy), "provenance": model.provenance,
        "topic_test_reserved": len(protocol["topic_test_indices"]), "topic_test_ambiguous_excluded": excluded,
        "topic_test": topic_metrics(model, final_rows), "spam_diagnostic": {},
        "category_validation_selection": read(directory / "category-selection.json"),
        "spam_validation_selection": read(directory / "spam-selection.json")}
    for origin in sorted({r["origin"] for r in diagnostic}):
        rows = [r for r in diagnostic if r["origin"] == origin]
        report["spam_diagnostic"][origin] = {"at_0_5": binary_metrics(model.spam, rows),
            "at_policy": binary_metrics(model.spam, rows, model.policy.spam_threshold)}
    report["spam_diagnostic"]["all_sources"] = {"at_0_5": binary_metrics(model.spam, diagnostic),
        "at_policy": binary_metrics(model.spam, diagnostic, model.policy.spam_threshold)}
    previous = Path("models/messages-v1.joblib")
    if previous.exists():
        old = load_model(previous)
        report["previous_synthetic_topic_on_same_test"] = topic_metrics(old, final_rows)
    private_previous = private_home() / "baseline-ai-v1.joblib"
    if private_previous.exists():
        report["previous_gmail_ai_topic_on_same_test"] = topic_metrics(load_model(private_previous), final_rows)
    # Per-message simulated outcomes remain private; no email body or subject is copied.
    from .message_model import classify_message
    outcomes = [{"id": r["id"], "expected_ai_category": r["category"],
                 **classify_message(model, r["subject"], r["body"])} for r in final_rows]
    save(directory / "thematic-test-predictions.json", {"source": "email_ai_labeled", "mailbox_changes": 0, "predictions": outcomes})
    save(directory / "final-evaluation.json", report)
    return {"topic_count": len(final_rows), "topic_agreement": report["topic_test"]["accuracy"],
            "diagnostic_spam_count": len(diagnostic), "human_gold_labels": False}


def simulate_recent(directory, collection_path):
    from .mail_input import content_group_keys
    from .message_model import classify_message
    directory, collection_path = private_path(directory), private_path(collection_path)
    freeze = verify_freeze(directory)
    if not (directory / "final-evaluation.json").exists():
        raise ValueError("Complete frozen thematic evaluation before recent-mail simulation")
    collection = read(collection_path)
    if collection.get("source") != "gmail_reviewed" or not isinstance(collection.get("records"), list):
        raise ValueError("Expected a read-only Gmail review collection")
    if not collection["records"]:
        raise ValueError("No recent messages collected")
    known = set()
    training = [r for r in read(directory / "development.json")["records"] if r["split"] == "train"]
    training += [r for r in read(directory / "category-development.json")["records"] if r["split"] == "train"]
    for row in training:
        minimized = {"subject": redact(row["subject"]), "body": redact(row["body"])}
        known.update((k, v.casefold()) for k, v in content_group_keys(minimized))
    model = load_model(directory / "messages-real-v2.joblib")
    results = []
    for row in collection["records"]:
        overlap = bool({(k, v.casefold()) for k, v in content_group_keys(row)} & known)
        results.append({"id": row["id"], "detected_training_overlap": overlap,
                        **classify_message(model, row["subject"], row["body"])})
    summary = {"count": len(results), "mailbox_changes": 0, "human_labels_available": False,
        "accuracy": None, "spam_precision": None, "category_accuracy": None,
        "model_sha256": freeze["model_sha256"], "collection_sha256": digest(collection_path),
        "detected_training_overlap_count": sum(r["detected_training_overlap"] for r in results),
        "labels": dict(Counter(r["label"] for r in results)),
        "category_candidates": dict(Counter(r["category_candidate"] or "none" for r in results)),
        "warning": "Unlabelled recent Gmail sample, not independent human accuracy or multi-provider validation. Predictions are suggestions only."}
    save(directory / "recent-reserved.json", collection)
    save(directory / "recent-predictions.json", {"summary": summary, "predictions": results})
    save(directory / "recent-summary.json", summary)
    save(private_home() / "active-model.json", {"model": str(directory / "messages-real-v2.joblib"),
        "report": str(directory / "final-evaluation.json"), "model_sha256": freeze["model_sha256"],
        "human_validation_pending": True, "category_suggestions_enabled": False}, overwrite=True)
    return summary


def known_training_keys(directory):
    from .mail_input import content_group_keys
    training = [r for r in read(directory / "development.json")["records"] if r["split"] == "train"]
    training += [r for r in read(directory / "category-development.json")["records"] if r["split"] == "train"]
    result = set()
    for row in training:
        minimized = {"subject": redact(row["subject"]), "body": redact(row["body"])}
        result.update((k, v.casefold()) for k, v in content_group_keys(minimized))
    return result


def finalize_review_workflow(directory):
    directory = private_path(directory)
    freeze = verify_freeze(directory)
    recent = read(directory / "recent-predictions.json")
    registry = {"model": str(directory / "messages-real-v2.joblib"),
        "report": str(directory / "final-evaluation.json"), "model_sha256": freeze["model_sha256"],
        "human_validation_pending": True, "review_only": True, "category_suggestions_enabled": False}
    save(private_home() / "active-model.json", registry, overwrite=True)
    # This operational layer never changes the frozen model or its raw experimental outputs.
    queue = [{"id": r["id"], "action": "review", "human_reviewed": False,
              "proposed_label": r["label"], "category_candidate": r["category_candidate"]}
             for r in recent["predictions"]]
    save(directory / "human-review-queue.json", {"mailbox_changes": 0, "review_required": len(queue), "records": queue})
    plan = {"human_review_complete": False, "independent_reference_available": False,
        "blind_public_review": str(private_home() / "email-category-review.json"),
        "blind_recent_review": str(directory / "recent-reserved.json"),
        "instructions": "Use the local review form on the original unlabelled collections. Do not copy AI predictions as human labels. Complete privacy review and exclusions; the evaluator never trains or changes thresholds.",
        "declared_review_flags_are_not_proof_of_independent_identity": True,
        "reviewer_identity_required_for_evaluation": True,
        "recent_collection_sha256": recent["summary"]["collection_sha256"],
        "recent_training_overlap_count": recent["summary"]["detected_training_overlap_count"]}
    save(directory / "human-validation-plan.json", plan)
    return {"active_model": registry["model"], "review_required": len(queue), "mailbox_changes": 0}


def evaluate_human(directory, collection_path, output_path, reviewer):
    from .mail_input import content_group_keys
    directory, collection_path, output_path = map(private_path, (directory, collection_path, output_path))
    verify_freeze(directory)
    if not reviewer.strip() or output_path.exists() or output_path in {collection_path, directory / "messages-real-v2.joblib"}:
        raise ValueError("Provide reviewer attribution and a new private report path")
    collection = read(collection_path)
    if collection.get("source") not in {"gmail_reviewed", "email_reviewed"} or not collection.get("records"):
        raise ValueError("Expected a human review collection")
    rows, seen, ids, duplicates = [], {}, set(), 0
    for row in collection["records"]:
        if any(type(row.get(k)) is not bool for k in ("exclude", "human_reviewed", "privacy_reviewed")):
            raise ValueError("Review flags must be explicit booleans")
        if row["exclude"]:
            continue
        if row.get("source") != collection["source"]:
            raise ValueError("Human reference must preserve its declared source")
        if not row["human_reviewed"] or not row["privacy_reviewed"]:
            raise ValueError("Human topic/spam and privacy review are still pending")
        if type(row.get("is_spam")) is not bool or (not row["is_spam"] and row.get("category") not in CATEGORIES):
            raise ValueError("Missing human spam/topic label")
        if row["is_spam"] and row.get("category") is not None:
            raise ValueError("Spam cannot have a thematic category")
        if not all(isinstance(row.get(k), str) for k in ("id", "group_id", "subject", "body")):
            raise ValueError("Invalid reviewed message")
        if row["id"] in ids:
            raise ValueError("Duplicate reviewed message id")
        ids.add(row["id"])
        key = canonical_text(row["subject"], row["body"])
        if not key:
            raise ValueError("Empty human-reviewed content")
        label = (row["is_spam"], row["category"])
        if key in seen:
            if seen[key] != label:
                raise ValueError("Conflicting human labels for duplicate content")
            duplicates += 1
            continue
        seen[key] = label
        rows.append(row)
    if not rows:
        raise ValueError("No reviewed messages included")
    model = load_model(directory / "messages-real-v2.joblib")
    probabilities = model.spam.predict_proba(text_rows(rows))[:, list(model.spam.classes_).index(True)]
    routed = [model.predict(r["subject"], r["body"]) for r in rows]
    truth = [r["is_spam"] for r in rows]
    accepted = np.array([r["label"] == "IA/SPAM" for r in routed], dtype=float)
    known = known_training_keys(directory)
    overlapping_groups = {r["group_id"] for r in rows
                          if {(k, v.casefold()) for k, v in content_group_keys(r)} & known}
    clean_indices = [i for i, r in enumerate(rows) if r["group_id"] not in overlapping_groups]
    report = {"source": collection["source"], "reviewer_attribution": reviewer.strip(),
        "human_review_declared": True, "independence_not_automatically_proven": True,
        "collection_sha256": digest(collection_path), "model_sha256": digest(directory / "messages-real-v2.joblib"),
        "count": len(rows), "exact_duplicates_not_counted": duplicates,
        "group_count": len({r["group_id"] for r in rows}),
        "spam_recall_assessable": any(truth), "category_assessable": any(not t for t in truth),
        "spam_at_0_5": evaluate_probabilities(truth, probabilities),
        "spam_suggestion_policy": evaluate_probabilities(truth, accepted),
        "topic_on_true_legitimate": topic_metrics(model, [r for r in rows if not r["is_spam"]]),
        "without_detected_training_overlap": {"count": len(clean_indices),
            "spam_at_0_5": evaluate_probabilities([truth[i] for i in clean_indices], probabilities[clean_indices]) if clean_indices else None,
            "topic_on_true_legitimate": topic_metrics(model, [rows[i] for i in clean_indices if not truth[i]])},
        "warning": "No fitting or threshold changes. A reviewed collection is not automatically representative of current providers, languages or users. Human flags and reviewer attribution are declarations, not verified identity."}
    save(output_path, report)
    return {"report": str(output_path), "count": len(rows), "spam_recall_assessable": any(truth)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=private_home() / "real-v2-20261105")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=Path("data/email/public-v1.json"))
    show = sub.add_parser("view")
    show.add_argument("--start", type=int, default=0)
    show.add_argument("--count", type=int, default=35)
    show.add_argument("--test", action="store_true")
    show.add_argument("--indices", type=int, nargs="+")
    annotation = sub.add_parser("annotate")
    annotation.add_argument("--labels", nargs="+", help="index|category-or-null|confidence|reason; no email contents")
    sub.add_parser("fit-spam")
    topics = sub.add_parser("fit-topics")
    topics.add_argument("--allow-private", action="store_true")
    topics.add_argument("--allow-ai-labels", action="store_true")
    sub.add_parser("evaluate")
    recent = sub.add_parser("simulate-recent")
    recent.add_argument("--collection", type=Path, required=True)
    sub.add_parser("finalize")
    sub.add_parser("status")
    human = sub.add_parser("evaluate-human")
    human.add_argument("--collection", type=Path, required=True)
    human.add_argument("--report", type=Path, required=True)
    human.add_argument("--reviewer", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.directory, args.dataset)
    elif args.command == "view":
        result = view(args.directory, args.start, args.count, test=args.test, indices=args.indices)
    elif args.command == "fit-spam":
        result = fit_spam(args.directory)
    elif args.command == "fit-topics":
        result = fit_topics_and_freeze(args.directory, allow_private=args.allow_private, allow_ai=args.allow_ai_labels)
    elif args.command == "evaluate":
        result = evaluate(args.directory)
    elif args.command == "simulate-recent":
        result = simulate_recent(args.directory, args.collection)
    elif args.command == "finalize":
        result = finalize_review_workflow(args.directory)
    elif args.command == "status":
        directory = private_path(args.directory)
        result = {"directory": str(directory), "files": sorted(p.name for p in directory.iterdir())}
    elif args.command == "evaluate-human":
        result = evaluate_human(args.directory, args.collection, args.report, args.reviewer)
    else:
        entries = json.load(sys.stdin) if not args.labels else [
            {"index": int(parts[0]), "category": None if parts[1] == "null" else parts[1],
             "confidence": float(parts[2]), "reason": parts[3]}
            for value in args.labels for parts in [value.split("|", 3)]]
        result = annotate(args.directory, entries)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
