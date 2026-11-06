import copy
from email.message import EmailMessage
import json
import mailbox

import numpy as np
import pytest

from gmail_classifier.baseline import load_model
from gmail_classifier.cli import main
from gmail_classifier.email_corpora import prepare_rows
from gmail_classifier.mail_input import iter_messages, parse_email
from gmail_classifier.message_model import read_categories, train_messages, validate_public


def test_parser_ignores_verdicts_attachments_and_prefers_plaintext():
    mail = EmailMessage()
    mail['Subject'] = 'Fatura disponível'
    mail['X-Spam-Status'] = 'privateverdict'
    mail.set_content('Corpo legítimo')
    mail.add_alternative('<p>htmlalternative</p><script>secretcode</script>', subtype='html')
    mail.add_attachment(b'attachmentsecret', maintype='text', subtype='plain', filename='note.txt')
    parsed = parse_email(mail.as_bytes())
    assert parsed['subject'] == 'Fatura disponível'
    assert 'Corpo legítimo' in parsed['body']
    assert all(s not in str(parsed) for s in ('privateverdict', 'htmlalternative', 'attachmentsecret'))


def test_html_and_unknown_charset_are_safe():
    raw = b'Subject: test\nContent-Type: text/html; charset=nonexistent\n\n<p>Hello</p><script>hidden</script>'
    result = parse_email(raw)
    assert 'Hello' in result['body'] and 'hidden' not in result['body']
    with pytest.raises(ValueError):
        parse_email(b'x' * 2_000_001)


def test_eml_mbox_and_plain_text(tmp_path):
    raw = b'Subject: Example\n\nMessage body'
    eml = tmp_path / 'example.eml'
    eml.write_bytes(raw)
    box_path = tmp_path / 'example.mbox'
    box = mailbox.mbox(box_path)
    box.add(raw)
    box.flush()
    box.close()
    assert list(iter_messages(eml)) == list(iter_messages(box_path))
    assert len(list(iter_messages(tmp_path))) == 1
    text = tmp_path / 'example.txt'
    text.write_text('Mensagem livre', encoding='utf-8')
    assert list(iter_messages(text))[0]['subject'] == ''


def test_public_split_is_reproducible_and_external_never_enters_training(public_dataset):
    validate_public(public_dataset)
    for row in public_dataset['records']:
        assert (row['split'] == 'external') == (row['source'] == 'spamassassin')
    broken = copy.deepcopy(public_dataset)
    broken['records'][-1]['split'] = 'train'
    with pytest.raises(ValueError, match='External'):
        validate_public(broken)


def test_overlap_templates_and_subject_threads_are_removed_before_splitting(public_dataset):
    rows = [{k: r[k] for k in ('subject', 'body', 'is_spam', 'source')} for r in public_dataset['records']]
    rows += [{'subject': 'Shared project discussion', 'body': 'A thread message with account 123', 'is_spam': False, 'source': 'enron'},
             {'subject': 'Re: Shared project discussion', 'body': 'Different reply from another mailbox', 'is_spam': False, 'source': 'spamassassin'},
             {'subject': 'other', 'body': 'offer https://abc.test/1 account 111', 'is_spam': True, 'source': 'enron'},
             {'subject': 'other', 'body': 'offer https://xyz.test/2 account 222', 'is_spam': True, 'source': 'spamassassin'}]
    prepared, stats = prepare_rows(rows)
    assert stats['enron_overlap_with_external_removed'] == 2
    assert all(r['subject'] != 'Shared project discussion' for r in prepared)
    assert prepare_rows(rows)[0] == prepared


def test_fit_isolates_both_stages_and_round_trips(public_dataset, records, tmp_path, capsys):
    data, cats = tmp_path / 'public.json', tmp_path / 'categories.json'
    data.write_text(json.dumps(public_dataset), encoding='utf-8')
    cats.write_text(json.dumps(records), encoding='utf-8')
    model_path, report_path = tmp_path / 'model.joblib', tmp_path / 'report.json'
    report = train_messages(data, cats, model_path, report_path)
    original = load_model(model_path)
    modified = copy.deepcopy(public_dataset)
    for row in modified['records']:
        if row['split'] != 'train':
            row['body'] += ' reservedspamtoken'
    for row in records:
        if row['split'] != 'train':
            row['body'] += ' reservedcategorytoken'
    data.write_text(json.dumps(modified), encoding='utf-8')
    cats.write_text(json.dumps(records), encoding='utf-8')
    train_messages(data, cats, model_path, report_path)
    loaded = load_model(model_path)
    assert 'reservedspamtoken' not in loaded.spam.named_steps['tfidf'].vocabulary_
    assert 'reservedcategorytoken' not in loaded.category.named_steps['tfidf'].vocabulary_
    np.testing.assert_array_equal(original.spam.named_steps['classifier'].coef_, loaded.spam.named_steps['classifier'].coef_)
    np.testing.assert_array_equal(original.category.named_steps['classifier'].coef_, loaded.category.named_steps['classifier'].coef_)
    assert 'Synthetic' in report['category_warning']
    assert report['spam']['external']['at_0_5']['count'] == 2
    main(['classify', '--model', str(model_path), '--text', 'meeting notes alpha'])
    assert 'spam_probability' in json.loads(capsys.readouterr().out)
    with pytest.raises(ValueError, match='overwrite'):
        train_messages(data, cats, data, report_path)


def test_category_data_requires_provenance_and_isolation(records, tmp_path):
    path = tmp_path / 'categories.json'
    rows = copy.deepcopy(records)
    rows[0]['source'] = 'email_reviewed'
    path.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        read_categories(path)
    for row in rows:
        row['source'] = 'email_ai_labeled'
    path.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError, match='allow-ai-labels'):
        read_categories(path, allow_private=True)
    rows = copy.deepcopy(records)
    rows[-1]['group_id'] = rows[0]['group_id']
    path.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError, match='group'):
        read_categories(path)


def test_import_review_never_marks_human_labels_and_supports_any_provider(tmp_path, capsys):
    from gmail_classifier.review_ui import load_collection
    from gmail_classifier.review import reviewed_records
    eml = tmp_path / 'message.eml'
    eml.write_bytes(b'Subject: A test message\nFrom: person@example.com\n\nContact person@example.com')
    output = tmp_path / 'review.json'
    main(['import-messages', '--input', str(eml), '--output', str(output)])
    assert json.loads(capsys.readouterr().out)['mailbox_changes'] == 0
    collection, revision = load_collection(output)
    assert collection['source'] == 'email_reviewed'
    row = collection['records'][0]
    assert row['human_reviewed'] is False and row['privacy_reviewed'] is False
    assert row['category'] is None and row['split'] is None
    assert 'person@example.com' not in row['body']
    with pytest.raises(ValueError, match='human'):
        reviewed_records(collection)
    with pytest.raises(SystemExit):
        main(['import-messages', '--input', str(eml), '--output', str(output)])


def test_tentative_category_does_not_override_review(model, monkeypatch):
    from gmail_classifier.message_model import classify_message
    monkeypatch.setattr(model.spam, 'predict_proba', lambda x: np.array([[0.4, 0.6]]))
    result = classify_message(model, 'Fatura', 'Pagamento do boleto')
    assert result['label'] == 'IA/Revisar' and result['category'] is None
    assert result['category_candidate'] in model.category.classes_


def test_message_channel_selects_sms_detector(model, monkeypatch, tmp_path, capsys):
    from gmail_classifier.baseline import save_model
    monkeypatch.setattr('gmail_classifier.spamshield.load_spamshield_model', lambda p: model.spam)
    path = tmp_path / 'model.joblib'
    save_model(model, path)
    main(['classify', '--model', str(path), '--channel', 'message', '--text', 'Fatura boleto'])
    result = json.loads(capsys.readouterr().out)
    assert result['model_provenance']['spam_source'] == 'M-Arjun/SpamShield-Datasets'


def test_category_review_sample_has_no_thematic_labels(public_dataset, tmp_path):
    from gmail_classifier.message_model import category_review_sample
    dataset, output = tmp_path / 'public.json', tmp_path / 'review.json'
    dataset.write_text(json.dumps(public_dataset), encoding='utf-8')
    assert category_review_sample(dataset, output, limit=5) == 5
    rows = json.loads(output.read_text(encoding='utf-8'))['records']
    assert len({r['group_id'] for r in rows}) == 5
    assert all(r['category'] is None and not r['human_reviewed'] and not r['privacy_reviewed'] for r in rows)


def test_human_category_contract_supports_partial_coverage(records, tmp_path):
    from gmail_classifier.review import reviewed_records
    rows = [dict(r, source='email_reviewed', human_reviewed=True, privacy_reviewed=True, exclude=False)
            for r in records if r['category'] in {'Financeiro', 'Comercial'}]
    collection = {'format_version': 1, 'source': 'email_reviewed', 'records': rows}
    prepared = reviewed_records(collection)
    path = tmp_path / 'category.json'
    path.write_text(json.dumps(prepared), encoding='utf-8')
    selected, source = read_categories(path, allow_private=True)
    assert source == 'email_reviewed' and len(selected) == len(rows)
