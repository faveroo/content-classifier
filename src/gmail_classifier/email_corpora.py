"""Acquire public mail corpora and prepare a source-held-out spam benchmark."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import tarfile
from urllib.request import urlopen
import zipfile

from .mail_input import MAX_BYTES, canonical_text, parse_email, template_text, thread_subject

SA_FILES = (
    "20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2", "20030228_spam.tar.bz2", "20050311_spam_2.tar.bz2",
)
ENRON_BASES = (
    "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/",
    "https://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/",
)
SA_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
ENRON_MIRROR = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"


def _fetch(url, target):
    with urlopen(url, timeout=25) as response:
        data = response.read(100_000_001)
    if len(data) > 100_000_000:
        raise ValueError("Archive exceeds download limit")
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(data)
    opener = zipfile.ZipFile if target.suffix == ".zip" else tarfile.open
    with opener(temporary):
        pass
    temporary.replace(target)


def download_corpora(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    sources = [(f"enron{i}.tar.gz", ENRON_BASES) for i in range(1, 7)]
    sources += [(name, (SA_BASE,)) for name in SA_FILES]
    for name, bases in sources:
        if name.startswith("enron") and (directory / "enron_spam_data.zip").exists():
            continue
        target = directory / name
        url = bases[0] + name
        if not target.exists():
            for base in bases:
                url = base + name
                try:
                    _fetch(url, target)
                    break
                except Exception:
                    continue
            else:
                if name.startswith("enron"):
                    target = directory / "enron_spam_data.zip"
                    try:
                        _fetch(ENRON_MIRROR, target)
                    except Exception:
                        raise RuntimeError("Enron source and CSV mirror unavailable; supply local archives") from None
                    manifest.append({"file": target.name, "url": ENRON_MIRROR,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
                    print(json.dumps({"downloaded": target.name, "fallback": "Enron-Spam CSV mirror by MWiechmann"}), flush=True)
                    continue
                raise RuntimeError(f"Could not download {name}; check network or supply local archives")
        with tarfile.open(target):
            pass
        manifest.append({"file": name, "url": url,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(json.dumps({"downloaded": name}), flush=True)
    mirror = directory / "enron_spam_data.zip"
    if mirror.exists() and not any(r["file"] == mirror.name for r in manifest):
        manifest.append({"file": mirror.name, "url": ENRON_MIRROR,
                         "sha256": hashlib.sha256(mirror.read_bytes()).hexdigest()})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_archives(directory):
    rows, skipped = [], 0
    csv.field_size_limit(MAX_BYTES)
    directory = Path(directory)
    names = list(SA_FILES)
    mirror = directory / "enron_spam_data.zip"
    if mirror.exists():
        with zipfile.ZipFile(mirror) as archive:
            csv_names = [n for n in archive.namelist() if n.endswith(".csv")]
            if len(csv_names) != 1 or archive.getinfo(csv_names[0]).file_size > 250_000_000:
                raise ValueError("Unexpected Enron CSV archive")
            with archive.open(csv_names[0]) as stream:
                for record in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")):
                    subject = record.get("Subject", "")
                    body = record.get("Message", "")
                    label = record.get("Spam/Ham")
                    if label not in {"spam", "ham"}:
                        raise ValueError("Invalid Enron CSV label")
                    if not (subject + body).strip():
                        skipped += 1
                        continue
                    rows.append({"subject": subject[:20_000], "body": body[:20_000],
                                 "is_spam": label == "spam", "source": "enron"})
    else:
        names = [f"enron{i}.tar.gz" for i in range(1, 7)] + names
    for name in names:
        with tarfile.open(directory / name) as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                label = next((p for p in parts[:-1] if p in {"ham", "spam", "spam_2", "easy_ham", "easy_ham_2", "hard_ham"}), None)
                if not member.isfile() or label is None or PurePosixPath(member.name).name.lower().startswith(("cmds", "readme")):
                    continue
                if member.size > MAX_BYTES:
                    skipped += 1
                    continue
                try:
                    with archive.extractfile(member) as stream:
                        parsed = parse_email(stream.read(MAX_BYTES + 1))
                except (ValueError, UnicodeError):
                    skipped += 1
                    continue
                rows.append({**parsed, "is_spam": label in {"spam", "spam_2"},
                             "source": "enron" if name.startswith("enron") else "spamassassin"})
    return rows, skipped


def prepare_rows(rows, seed=42, *, training_source="enron", external_source="spamassassin"):
    """Group content/templates/subjects BEFORE splitting. Reserve SpamAssassin entirely."""
    rows = [dict(r) for r in rows]
    if any(r["source"] not in {training_source, external_source} for r in rows):
        raise ValueError("Unexpected source during grouping")
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys, exact_labels = {}, defaultdict(set)
    for i, row in enumerate(rows):
        text = canonical_text(row["subject"], row["body"])
        exact_labels[text].add(row["is_spam"])
        subject = thread_subject(row["subject"])
        candidates = [("exact", text), ("template", template_text(text))]
        candidates.extend(("thread_id", value) for value in row.get("_thread_links", ()))
        body_template = template_text(row["body"])
        if len(body_template) >= 80:
            candidates.append(("body_template", body_template))
        if len(subject) >= 12 and len(subject.split()) >= 3:
            candidates.append(("subject", subject))
        for key in candidates:
            if key in keys:
                parent[root(i)] = root(keys[key])
            else:
                keys[key] = i
    conflicting = {root(i) for i, row in enumerate(rows)
                   if len(exact_labels[canonical_text(row["subject"], row["body"])]) > 1}
    external_groups = {root(i) for i, row in enumerate(rows) if row["source"] == external_source}
    kept, seen, stats = [], set(), Counter()
    for i, row in enumerate(rows):
        if root(i) in conflicting:
            stats["conflicting_group_rows_removed"] += 1
            continue
        if row["source"] == training_source and root(i) in external_groups:
            stats[f"{training_source}_overlap_with_external_removed"] += 1
            continue
        text = canonical_text(row["subject"], row["body"])
        if text in seen:
            stats["exact_duplicates_removed"] += 1
            continue
        seen.add(text)
        row.update(id=hashlib.sha256(text.encode()).hexdigest(), group_id=str(root(i)))
        row.pop("_thread_links", None)
        kept.append(row)
    groups = defaultdict(list)
    for row in kept:
        if row["source"] == external_source:
            row["split"] = "external"
        else:
            groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        strata[tuple(sorted({r["is_spam"] for r in members}))].append(group)
    rng = random.Random(seed)
    for signature in sorted(strata):
        ids = sorted(strata[signature])
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * .15)) if len(ids) >= 3 else 0
        n_validation = max(1, round(len(ids) * .15)) if len(ids) >= 3 else 0
        for i, group in enumerate(ids):
            split = "test" if i < n_test else "validation" if i < n_test + n_validation else "train"
            for row in groups[group]:
                row["split"] = split
    split_names = ("train", "validation", "test") + (("external",) if external_source else ())
    for split in split_names:
        if {r["is_spam"] for r in kept if r["split"] == split} != {False, True}:
            raise ValueError(f"{split} needs independent groups of both spam classes")
    return kept, dict(stats)


def prepare_corpora(directory, output):
    directory, output = Path(directory), Path(output)
    rows, skipped = read_archives(directory)
    prepared, stats = prepare_rows(rows)
    manifest = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(directory.iterdir()) if p.name.endswith((".tar.gz", ".tar.bz2", ".zip"))]
    dataset = {"format_version": 1, "source": "public_email", "seed": 42,
               "input_count": len(rows), "parsing_skipped": skipped, "preparation": stats,
               "archives": manifest, "records": prepared}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.resolve() in {p.resolve() for p in directory.iterdir()}:
        raise ValueError("Choose a new dataset output path")
    output.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return {"prepared": len(prepared), "splits": dict(Counter(r["split"] for r in prepared)), **stats}
