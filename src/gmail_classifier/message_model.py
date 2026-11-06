"""Independent spam/category training and offline inference for any provider."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

from .baseline import Baseline, pipeline, save_model
from .dataset import CATEGORIES, SPLITS
from .mail_input import canonical_text, template_text, thread_subject
from .policy import Policy
from .private_storage import private_path


def read_categories(path, *, allow_private=False, allow_ai=False):
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Category dataset must be a non-empty JSON array")
    sources = {r.get("source") for r in records if isinstance(r, dict)}
    if len(sources) != 1:
        raise ValueError("Keep category label provenance separate")
    source = next(iter(sources))
    if source not in {"synthetic", "email_reviewed", "gmail_reviewed", "email_ai_labeled", "gmail_ai_labeled"}:
        raise ValueError("Unsupported category source")
    if source != "synthetic":
        if not allow_private:
            raise ValueError("Real category datasets require --allow-private")
        private_path(path)
    if source.endswith("ai_labeled") and not allow_ai:
        raise ValueError("AI categories require --allow-ai-labels")
    seen, seen_text, groups, templates, subjects = set(), set(), {}, {}, {}
    selected = []
    for row in records:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("id", "group_id", "subject", "body", "split", "source")):
            raise ValueError("Invalid category record")
        if (row["split"] not in SPLITS or not row["id"].strip() or not row["group_id"].strip()
                or type(row.get("is_spam")) is not bool):
            raise ValueError("Invalid category identifiers/split/spam label")
        text = canonical_text(row["subject"], row["body"])
        if not text or row["id"] in seen or text in seen_text:
            raise ValueError("Empty category text or duplicate id")
        seen.add(row["id"])
        seen_text.add(text)
        signature = template_text(text)
        if signature in templates and templates[signature] != row["split"]:
            raise ValueError("Category template crosses split boundary")
        templates[signature] = row["split"]
        subject = thread_subject(row["subject"])
        if len(subject) >= 12 and len(subject.split()) >= 3:
            if subject in subjects and subjects[subject] != row["split"]:
                raise ValueError("Category subject thread crosses split boundary")
            subjects[subject] = row["split"]
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError("Category group crosses split boundary")
        groups[row["group_id"]] = row["split"]
        if row["is_spam"]:
            if row.get("category") is not None:
                raise ValueError("Spam must not have a thematic category")
            continue
        if row.get("category") not in CATEGORIES:
            raise ValueError("Unsupported thematic category")
        selected.append(row)
    trained = {r["category"] for r in selected if r["split"] == "train"}
    if len(trained) < 2:
        raise ValueError("Category training requires at least two categories")
    for split in SPLITS:
        rows = [r for r in selected if r["split"] == split]
        if not rows or any(r["category"] not in trained for r in rows):
            raise ValueError("Every category split needs examples covered in training")
    return selected, source


def text_rows(rows):
    return [canonical_text(r["subject"], r["body"]) for r in rows]


def evaluate_probabilities(expected, probabilities, threshold=.5):
    expected = np.asarray(expected, dtype=int)
    guess = np.asarray(probabilities) >= threshold
    matrix = confusion_matrix(expected, guess, labels=[0, 1]).tolist()
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {"count": len(expected), "threshold": threshold,
            "accuracy": float((guess == expected).mean()), "labels": ["ham", "spam"],
            "confusion_matrix": matrix, "false_positives": fp, "false_negatives": fn,
            "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "spam_precision": tp / (tp + fp) if tp + fp else None,
            "spam_recall": tp / (tp + fn) if tp + fn else None,
            "classification_report": classification_report(expected, guess, labels=[0, 1],
                target_names=["ham", "spam"], output_dict=True, zero_division=0)}


def validate_public(dataset):
    if dataset.get("source") != "public_email" or dataset.get("format_version") != 1:
        raise ValueError("Expected prepared public e-mail dataset")
    rows = dataset.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty public e-mail dataset")
    training_source = dataset.get("training_source", "enron")
    external_source = dataset.get("external_source", "spamassassin")
    if training_source not in {"enron", "kaggle_isuranga"} or external_source not in {None, "spamassassin"}:
        raise ValueError("Unsupported public email sources")
    split_names = (*SPLITS, "external") if external_source else SPLITS
    seen, groups, templates, subjects = set(), {}, {}, {}
    for row in rows:
        if not all(isinstance(row.get(k), str) for k in ("id", "group_id", "subject", "body", "source", "split")):
            raise ValueError("Invalid public email record")
        split = row["split"]
        if type(row.get("is_spam")) is not bool or split not in split_names:
            raise ValueError("Invalid public email split/label")
        if row["source"] != (external_source if split == "external" else training_source):
            raise ValueError("External corpus must stay out of training and internal splits")
        text = canonical_text(row["subject"], row["body"])
        if not text or text in seen:
            raise ValueError("Empty or duplicate public text")
        seen.add(text)
        checks = [(groups, row["group_id"]), (templates, ("full", template_text(text)))]
        if len(body_template := template_text(row["body"])) >= 80:
            checks.append((templates, ("body", body_template)))
        for mapping, key in checks:
            if key in mapping and mapping[key] != split:
                raise ValueError("Public group/template crosses split boundary")
            mapping[key] = split
        subject = thread_subject(row["subject"])
        if len(subject) >= 12 and len(subject.split()) >= 3:
            if subject in subjects and subjects[subject] != split:
                raise ValueError("Public thread crosses split boundary")
            subjects[subject] = split
    for split in split_names:
        if {r["is_spam"] for r in rows if r["split"] == split} != {False, True}:
            raise ValueError("Each spam evaluation split needs both labels")
    return rows


def train_messages(dataset_path, category_path, model_path, report_path, *, allow_private=False,
                   allow_ai=False, sms_reference=None, email_reference=None):
    dataset_path, category_path, model_path, report_path = map(Path,
        (dataset_path, category_path, model_path, report_path))
    protected = {dataset_path.resolve(), category_path.resolve()}
    if sms_reference:
        protected.add(Path(sms_reference).resolve())
    if email_reference:
        protected.add(Path(email_reference).resolve())
    if model_path.resolve() == report_path.resolve() or {model_path.resolve(), report_path.resolve()} & protected:
        raise ValueError("Outputs must differ and must not overwrite inputs")
    # Inspect provenance before opening real category content or writing outputs.
    categories, category_source = read_categories(category_path, allow_private=allow_private, allow_ai=allow_ai)
    if category_source != "synthetic":
        private_path(model_path)
        private_path(report_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = validate_public(dataset)
    spam_train = [r for r in rows if r["split"] == "train"]
    category_train = [r for r in categories if r["split"] == "train"]
    spam, category = pipeline(), pipeline()
    spam.fit(text_rows(spam_train), [r["is_spam"] for r in spam_train])
    category.fit(text_rows(category_train), [r["category"] for r in category_train])
    provenance = {"spam_source": dataset.get("corpus", "Enron-Spam"), "category_source": category_source,
                  "human_gold_labels": category_source in {"email_reviewed", "gmail_reviewed"}}
    model = Baseline(spam, category, Policy(), provenance)
    report = {"format_version": 1, "source": "public_email", "sklearn_version": sklearn.__version__,
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "category_dataset_sha256": hashlib.sha256(category_path.read_bytes()).hexdigest(),
              "model_provenance": provenance, "policy": asdict(model.policy), "seed": 42,
              "preparation": {k: v for k, v in dataset.items() if k != "records"},
              "warning": dataset.get("warning", "Historical mostly English email; not current multilingual production validation. Exact/template grouping cannot identify all paraphrases."),
              "category_warning": ("Synthetic smoke test; not real-world category quality" if category_source == "synthetic" else
                  "AI label agreement, not independent human accuracy" if category_source.endswith("ai_labeled") else
                  "Reviewed sample; generalization still requires external category evaluation"),
              "policy_metric_note": "At policy: binary spam suggestion versus no spam suggestion; non-suggestions include review, not just accepted ham.",
              "spam": {}, "category": {}}
    sms_model = None
    email_model = None
    if email_reference:
        from .baseline import load_model
        email_model = load_model(email_reference).spam
        report["email_comparison_warning"] = "The full comparison can include prior training overlap. The filtered subset excludes detected groups overlapping the supplied reference dataset train, when that audit is available; this is heuristic, not proof of independence."
    if sms_reference:
        from .spamshield import load_spamshield_model
        sms_model = load_spamshield_model(sms_reference)
        report["sms_comparison_warning"] = "Same email evaluation texts; SpamShield's underlying corpus overlap is unknown, so this is not an assured independent benchmark for that model."
    split_names = (*SPLITS, "external") if dataset.get("external_source", "spamassassin") else SPLITS
    for split in split_names:
        members = [r for r in rows if r["split"] == split]
        texts = text_rows(members)
        probability = spam.predict_proba(texts)[:, list(spam.classes_).index(True)]
        known = np.asarray(spam.named_steps["tfidf"].transform(texts).getnnz(axis=1)) > 0
        report["spam"][split] = {"at_0_5": evaluate_probabilities([r["is_spam"] for r in members], probability),
            "at_policy": evaluate_probabilities([r["is_spam"] for r in members], np.where(known, probability, 0.0), model.policy.spam_threshold),
            "group_count": len({r["group_id"] for r in members}),
            "distribution": dict(Counter("spam" if r["is_spam"] else "ham" for r in members))}
        if sms_model is not None and split != "train":
            sms_probability = sms_model.predict_proba(texts)[:, list(sms_model.classes_).index(1)]
            report["spam"][split]["sms_reference_at_0_5"] = evaluate_probabilities([r["is_spam"] for r in members], sms_probability)
        if email_model is not None and split != "train":
            old_probability = email_model.predict_proba(texts)[:, list(email_model.classes_).index(True)]
            report["spam"][split]["email_reference_at_0_5"] = evaluate_probabilities([r["is_spam"] for r in members], old_probability)
            if "reference_overlap_audit" in dataset:
                clean = [i for i, r in enumerate(members) if r.get("reference_training_overlap") is False]
                report["spam"][split]["comparison_without_detected_prior_train_overlap"] = {
                    "count": len(clean),
                    "new_model": evaluate_probabilities([members[i]["is_spam"] for i in clean], probability[clean]) if clean else None,
                    "reference_model": evaluate_probabilities([members[i]["is_spam"] for i in clean], old_probability[clean]) if clean else None}
        if all(isinstance(r.get("language"), str) for r in members):
            report["spam"][split]["by_language"] = {}
            for language in sorted({r["language"] for r in members}):
                indices = [i for i, r in enumerate(members) if r["language"] == language]
                report["spam"][split]["by_language"][language] = evaluate_probabilities([members[i]["is_spam"] for i in indices], probability[indices])
    for split in SPLITS:
        members = [r for r in categories if r["split"] == split]
        texts = text_rows(members)
        guesses = category.predict(texts)
        expected = [r["category"] for r in members]
        routed = [model.predict(r["subject"], r["body"]) for r in members]
        report["category"][split] = {"count": len(members), "distribution": dict(Counter(expected)),
            "labels": list(CATEGORIES), "accuracy": float(np.mean(guesses == np.array(expected))),
            "classification_report": classification_report(expected, guesses, labels=list(CATEGORIES), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(expected, guesses, labels=list(CATEGORIES)).tolist(),
            "trained_categories": sorted(category.classes_.tolist()),
            "missing_training_categories": sorted(set(CATEGORIES) - set(category.classes_)),
            "review_rate": sum(r["action"] == "review" for r in routed) / len(routed),
            "end_to_end_correct_label_rate": sum(r["label"] == f"IA/{e}" for r, e in zip(routed, expected)) / len(expected)}
    save_model(model, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def import_for_review(input_path, output_path):
    from .mail_input import iter_messages
    from .text_privacy import redact
    from .private_storage import write_private_json
    import secrets
    output_path = private_path(output_path)
    if output_path.exists() or output_path == Path(input_path).resolve():
        raise ValueError("Choose a new private review output path")
    rows = []
    groups = {}
    for message in iter_messages(input_path):
        signature = thread_subject(message["subject"]) or template_text(message["body"])
        group = groups.setdefault(signature, "g-" + secrets.token_hex(12))
        rows.append({"id": "m-" + secrets.token_hex(12), "group_id": group,
                     "subject": redact(message["subject"]), "body": redact(message["body"]),
                     "is_spam": None, "category": None, "source": "email_reviewed", "split": None,
                     "human_reviewed": False, "privacy_reviewed": False, "exclude": False})
    if not rows:
        raise ValueError("No messages found")
    write_private_json(output_path, {"format_version": 1, "source": "email_reviewed", "records": rows})
    return len(rows)


def category_review_sample(dataset_path, output_path, limit=180):
    """A deterministic, unlabelled real-mail sample; no pseudo human approvals."""
    import random
    import secrets
    from .text_privacy import redact
    from .private_storage import write_private_json
    output_path = private_path(output_path)
    if output_path.exists() or not 1 <= limit <= 1000:
        raise ValueError("Choose a new private review file and a limit from 1 to 1000")
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    rows = validate_public(dataset)
    # Only Enron ham: the external spam benchmark is not repurposed for category fit.
    candidates = [r for r in rows if r["source"] == "enron" and not r["is_spam"]]
    random.Random(42).shuffle(candidates)
    selected, groups = [], set()
    for row in candidates:
        if row["group_id"] in groups:
            continue
        groups.add(row["group_id"])
        selected.append({"id": "m-" + secrets.token_hex(12), "group_id": "g-" + secrets.token_hex(12),
            "subject": redact(row["subject"]), "body": redact(row["body"]),
            "is_spam": False, "category": None, "source": "email_reviewed", "split": row["split"],
            "human_reviewed": False, "privacy_reviewed": False, "exclude": False})
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("No eligible legitimate email groups")
    write_private_json(output_path, {"format_version": 1, "source": "email_reviewed", "records": selected})
    return len(selected)


def classify_message(model, subject, body, *, conditional_topic=False):
    """Expose a tentative topic without treating it as an accepted policy label."""
    result = model.predict(subject, body)
    result.update(category_candidate=None, category_candidate_confidence=None)
    if conditional_topic or result["spam_probability"] < model.policy.spam_threshold:
        text = canonical_text(subject, body)
        if model.category.named_steps["tfidf"].transform([text]).nnz:
            probabilities = model.category.predict_proba([text])[0]
            index = int(probabilities.argmax())
            result.update(category_candidate=str(model.category.classes_[index]),
                          category_candidate_confidence=float(probabilities[index]))
            if conditional_topic and result["spam_probability"] >= model.policy.spam_threshold:
                result["category_candidate_condition"] = "if_legitimate"
    return result


def evaluate_external(model_path, dataset_path, training_dataset_path, *, reference_model=None):
    """Post-fit diagnostic on a prior external corpus; does not select parameters."""
    from .baseline import load_model
    from .mail_input import content_group_keys
    model = load_model(model_path)
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    training_dataset = json.loads(Path(training_dataset_path).read_text(encoding="utf-8"))
    external = [r for r in validate_public(dataset) if r["split"] == "external"]
    training = [r for r in validate_public(training_dataset) if r["split"] == "train"]
    if not external:
        raise ValueError("The diagnostic dataset needs a reserved external split")
    keys = set()
    for row in training:
        keys.update(content_group_keys(row))
    excluded = {r["group_id"] for r in external if content_group_keys(r) & keys}
    selected = [r for r in external if r["group_id"] not in excluded]
    if {r["is_spam"] for r in selected} != {False, True}:
        raise ValueError("External diagnostic needs both classes after overlap removal")
    texts = text_rows(selected)
    truth = [r["is_spam"] for r in selected]
    probabilities = model.spam.predict_proba(texts)[:, list(model.spam.classes_).index(True)]
    known = np.asarray(model.spam.named_steps["tfidf"].transform(texts).getnnz(axis=1)) > 0
    report = {"source": "external_email_diagnostic", "model_provenance": model.provenance,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "training_dataset_sha256": hashlib.sha256(Path(training_dataset_path).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "warning": "Reused historical external corpus; no fitting, calibration or parameter selection. Overlap filtering is heuristic. No thematic category evaluation.",
        "input_count": len(external), "overlapping_groups_removed": len(excluded),
        "overlapping_rows_removed": len(external) - len(selected),
        "at_0_5": evaluate_probabilities(truth, probabilities),
        "at_policy": evaluate_probabilities(truth, np.where(known, probabilities, 0.0), model.policy.spam_threshold)}
    if reference_model:
        old = load_model(reference_model).spam
        old_probability = old.predict_proba(texts)[:, list(old.classes_).index(True)]
        report["email_reference_at_0_5"] = evaluate_probabilities(truth, old_probability)
    return report
