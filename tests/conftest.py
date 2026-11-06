from pathlib import Path

import pytest

from gmail_classifier.baseline import train
from gmail_classifier.dataset import load_dataset


@pytest.fixture
def records():
    return load_dataset(Path(__file__).resolve().parents[1] / "data/examples/synthetic-v1.json")


@pytest.fixture
def model(records):
    return train(records)


@pytest.fixture
def public_dataset():
    from gmail_classifier.email_corpora import prepare_rows
    rows = []
    # Alphabetic differences survive the normalization of numbers in templates.
    words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
             'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa']
    for word in words:
        for spam in (False, True):
            rows.append({'subject': word, 'body': ('prize offer ' if spam else 'meeting notes ') + word,
                         'is_spam': spam, 'source': 'enron'})
    rows += [{'subject': 'outside', 'body': 'external cash giveaway', 'is_spam': True, 'source': 'spamassassin'},
             {'subject': 'outside', 'body': 'external family dinner', 'is_spam': False, 'source': 'spamassassin'}]
    prepared, _ = prepare_rows(rows)
    return {'format_version': 1, 'source': 'public_email', 'records': prepared}
