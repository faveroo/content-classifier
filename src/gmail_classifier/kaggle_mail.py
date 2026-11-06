"""Separate, reproducible experiment for isuranga's Kaggle mail corpus."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

from .email_corpora import prepare_rows
from .mail_input import HTMLText, MAX_TEXT, content_group_keys

DATASET_ID = "isuranga/spam-genuine-mail-contents-dataset"
DATASET_URL = "https://www.kaggle.com/datasets/" + DATASET_ID
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/" + DATASET_ID
METADATA_URL = "https://www.kaggle.com/api/v1/datasets/view/" + DATASET_ID


def download_dataset(directory):
    """Download public files only; never reads or prints local Kaggle credentials."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dataset.zip"
    try:
        if not target.exists():
            request = Request(DOWNLOAD_URL, headers={"User-Agent": "gmail-email-classifier/0.1"})
            with urlopen(request, timeout=45) as response:
                raw = response.read(100_000_001)
            if len(raw) > 100_000_000:
                raise ValueError("Download exceeds 100 MB")
            partial = directory / "dataset.zip.partial"
            partial.write_bytes(raw)
            with zipfile.ZipFile(partial):
                pass
            partial.replace(target)
        with zipfile.ZipFile(target) as archive:
            files = [{"file": item.filename, "bytes": item.file_size} for item in archive.infolist()]
    except Exception:
        raise RuntimeError("Kaggle download unavailable. Check network/access, or download the dataset ZIP from its page and save it as dataset.zip in the selected directory.") from None
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        try:
            request = Request(METADATA_URL, headers={"User-Agent": "gmail-email-classifier/0.1"})
            with urlopen(request, timeout=20) as response:
                metadata = json.loads(response.read(5_000_000))
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # Metadata unavailability does not relabel the license or block local data use.
    manifest = {"source": DATASET_ID, "url": DATASET_URL,
                "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_rows(archive_path):
    rows, skipped = [], 0
    csv.field_size_limit(2_000_000)
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if names != ["email_dataset_100k.csv"] or archive.getinfo(names[0]).file_size > 200_000_000:
            raise ValueError("Unexpected Kaggle mail archive")
        with archive.open(names[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            if not {"subject", "body_plain", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("Kaggle CSV requires subject/body_plain/label")
            for record in reader:
                if record.get("label") not in {"0", "1"}:
                    raise ValueError("Kaggle label must be exactly 0 or 1")
                subject = (record.get("subject") or "").strip()[:MAX_TEXT]
                body = (record.get("body_plain") or "").strip()[:MAX_TEXT]
                if not body and record.get("body_html"):
                    parser = HTMLText()
                    parser.feed(record["body_html"][:2_000_000])
                    body = " ".join(parser.parts).strip()[:MAX_TEXT]
                if not (subject + body).strip():
                    skipped += 1
                    continue
                # Explicit allowlist: raw_text, synthetic spam scores, authentication,
                # addresses and other metadata are NEVER used as model features.
                rows.append({"subject": subject, "body": body, "is_spam": record["label"] == "1",
                             "source": "kaggle_isuranga", "language": (record.get("language") or "unknown").strip(),
                             "_thread_links": tuple(hashlib.sha256(record[key].strip().encode()).hexdigest()
                                 for key in ("message_id", "in_reply_to") if record.get(key, "").strip())})
    return rows, skipped


def prepare_dataset(directory, output, *, reference_dataset=None):
    directory, output = Path(directory), Path(output)
    if output.exists() or output.resolve() in {p.resolve() for p in directory.iterdir()}:
        raise ValueError("Choose a new prepared dataset output path")
    rows, skipped = read_rows(directory / "dataset.zip")
    prepared, stats = prepare_rows(rows, training_source="kaggle_isuranga", external_source=None)
    dataset = {"format_version": 1, "source": "public_email", "corpus": DATASET_ID,
               "training_source": "kaggle_isuranga", "external_source": None,
               "synthetic": True, "seed": 42, "input_count": len(rows), "parsing_skipped": skipped,
               "preparation": stats, "archive_sha256": hashlib.sha256((directory / "dataset.zip").read_bytes()).hexdigest(),
               "warning": "Synthetic generated email benchmark, not real-world accuracy. Language tags are publisher declarations. Grouping heuristics cannot isolate all generation patterns.",
               "features": ["subject", "body_plain (HTML fallback only when plain is empty)"],
               "records": prepared}
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    dataset["attribution"] = {"author": metadata.get("creatorName") or "isuranga", "url": DATASET_URL,
        "license_declared_by_publisher": metadata.get("licenseName", "not verified"),
        "version": metadata.get("currentVersionNumber"), "last_updated": metadata.get("lastUpdated")}
    if reference_dataset:
        from .message_model import validate_public
        reference_path = Path(reference_dataset)
        reference_rows = validate_public(json.loads(reference_path.read_text(encoding="utf-8")))
        prior_keys = set()
        for row in reference_rows:
            if row["split"] == "train":
                prior_keys.update(content_group_keys(row))
        overlapping_groups = {r["group_id"] for r in prepared if content_group_keys(r) & prior_keys}
        for row in prepared:
            row["reference_training_overlap"] = row["group_id"] in overlapping_groups
        dataset["reference_overlap_audit"] = {
            "reference_dataset_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "overlapping_groups": len(overlapping_groups),
            "overlapping_rows_by_split": dict(Counter(r["split"] for r in prepared if r["reference_training_overlap"]))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return {"prepared": len(prepared), "groups": len({r["group_id"] for r in prepared}),
            "splits": dict(Counter(r["split"] for r in prepared)), "preparation": stats,
            "reference_overlap_audit": dataset.get("reference_overlap_audit")}
