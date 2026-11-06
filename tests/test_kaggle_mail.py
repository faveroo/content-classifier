import copy
import csv
import io
import json
import zipfile

import pytest

from gmail_classifier.baseline import load_model, save_model
from gmail_classifier.kaggle_mail import download_dataset, prepare_dataset, read_rows
from gmail_classifier.message_model import train_messages, validate_public


def write_archive(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(directory / 'dataset.zip', 'w') as archive:
        archive.writestr('email_dataset_100k.csv', text.getvalue())


def test_features_exclude_score_raw_headers_and_addresses(tmp_path):
    write_archive(tmp_path, [{'subject': 'Invoice', 'body_plain': 'A normal invoice', 'label': '0',
        'raw_text': 'poisonrawheader', 'x_spam_score': 'poisonscore', 'from_address': 'poisonsender',
        'language': 'en', 'message_id': 'opaque-id', 'in_reply_to': 'other-id'}])
    rows, skipped = read_rows(tmp_path / 'dataset.zip')
    assert skipped == 0
    assert set(rows[0]) == {'subject', 'body', 'is_spam', 'source', 'language', '_thread_links'}
    assert all('poison' not in str(v) for v in rows[0].values())
    assert len(rows[0]['_thread_links']) == 2
    assert 'opaque-id' not in str(rows)


def test_html_fallback_and_strict_labels(tmp_path):
    base = {'subject': '', 'body_plain': '', 'body_html': '<p>Hello</p><script>secret</script>', 'label': '1'}
    write_archive(tmp_path, [base])
    rows, _ = read_rows(tmp_path / 'dataset.zip')
    assert 'Hello' in rows[0]['body'] and 'secret' not in rows[0]['body']
    write_archive(tmp_path, [dict(base, label='1.0')])
    with pytest.raises(ValueError, match='exactly'):
        read_rows(tmp_path / 'dataset.zip')


def test_prepare_has_three_isolated_splits_and_reference_overlap_audit(public_dataset, tmp_path):
    # Reuse independently worded fixture messages, with a known duplicate for removal.
    input_rows = [{'subject': r['subject'], 'body_plain': r['body'], 'label': str(int(r['is_spam'])), 'language': 'en'}
                  for r in public_dataset['records'] if r['source'] == 'enron']
    input_rows.append(dict(input_rows[0]))
    write_archive(tmp_path, input_rows)
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps(public_dataset), encoding='utf-8')
    output = tmp_path / 'prepared.json'
    summary = prepare_dataset(tmp_path, output, reference_dataset=reference)
    data = json.loads(output.read_text(encoding='utf-8'))
    rows = validate_public(data)
    assert set(summary['splits']) == {'train', 'validation', 'test'}
    assert summary['preparation']['exact_duplicates_removed'] == 1
    assert summary['reference_overlap_audit']['overlapping_groups'] > 0
    assert data['synthetic'] is True
    assert all(r['source'] == 'kaggle_isuranga' for r in rows)
    assert not any('_thread_links' in r for r in rows)
    with pytest.raises(ValueError, match='new prepared'):
        prepare_dataset(tmp_path, output)
    broken = copy.deepcopy(data)
    broken['records'][0]['split'] = 'external'
    with pytest.raises(ValueError):
        validate_public(broken)


def test_kaggle_train_preserves_categories_and_compares_old_model(public_dataset, records, model, tmp_path):
    from gmail_classifier.kaggle_mail import DATASET_ID
    data = copy.deepcopy(public_dataset)
    data.update(training_source='kaggle_isuranga', external_source=None, corpus=DATASET_ID,
                reference_overlap_audit={'overlapping_groups': 0})
    data['records'] = [r for r in data['records'] if r['split'] != 'external']
    for row in data['records']:
        row.update(source='kaggle_isuranga', language='en', reference_training_overlap=False)
    dataset, categories = tmp_path / 'data.json', tmp_path / 'cats.json'
    dataset.write_text(json.dumps(data), encoding='utf-8')
    categories.write_text(json.dumps(records), encoding='utf-8')
    old_model, artifact, report_path = tmp_path / 'old.joblib', tmp_path / 'new.joblib', tmp_path / 'report.json'
    save_model(model, old_model)
    report = train_messages(dataset, categories, artifact, report_path, email_reference=old_model)
    assert report['model_provenance']['spam_source'] == DATASET_ID
    assert report['model_provenance']['category_source'] == 'synthetic'
    assert set(report['spam']) == {'train', 'validation', 'test'}
    for split in ('validation', 'test'):
        metrics = report['spam'][split]
        assert metrics['comparison_without_detected_prior_train_overlap']['count'] == metrics['at_0_5']['count']
        assert metrics['by_language']['en']['count'] == metrics['at_0_5']['count']
    assert load_model(artifact).provenance['spam_source'] == DATASET_ID
    with pytest.raises(ValueError, match='overwrite'):
        train_messages(dataset, categories, old_model, report_path, email_reference=old_model)


def test_download_does_not_publish_non_zip_responses(tmp_path, monkeypatch):
    monkeypatch.setattr('gmail_classifier.kaggle_mail.urlopen', lambda *a, **k: io.BytesIO(b'<html>Login required</html>'))
    with pytest.raises(RuntimeError, match='Kaggle download'):
        download_dataset(tmp_path)
    assert not (tmp_path / 'dataset.zip').exists()


def test_external_diagnostic_never_fits_and_checks_cross_corpus_overlap(public_dataset, model, tmp_path, monkeypatch):
    from gmail_classifier.message_model import evaluate_external
    data, training_path, artifact = tmp_path / 'external.json', tmp_path / 'training.json', tmp_path / 'model.joblib'
    data.write_text(json.dumps(public_dataset), encoding='utf-8')
    training = copy.deepcopy(public_dataset)
    training.update(training_source='kaggle_isuranga', external_source=None)
    training['records'] = [r for r in training['records'] if r['split'] != 'external']
    for row in training['records']:
        row['source'] = 'kaggle_isuranga'
    training_path.write_text(json.dumps(training), encoding='utf-8')
    save_model(model, artifact)
    monkeypatch.setattr('sklearn.pipeline.Pipeline.fit', lambda *a, **k: pytest.fail('Diagnostic must never fit'))
    report = evaluate_external(artifact, data, training_path, reference_model=artifact)
    assert report['input_count'] == report['at_0_5']['count'] == 2
    assert report['overlapping_rows_removed'] == 0
    outside = public_dataset['records'][-1]
    member = next(r for r in training['records'] if r['split'] == 'train' and r['is_spam'] == outside['is_spam'])
    member.update(subject=outside['subject'], body=outside['body'])
    training_path.write_text(json.dumps(training), encoding='utf-8')
    with pytest.raises(ValueError, match='both classes'):
        evaluate_external(artifact, data, training_path)
