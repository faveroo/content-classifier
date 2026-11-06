import copy
import json

import joblib
import numpy as np
import pytest

from gmail_classifier.experiment_v2 import (annotate, candidate, choose_spam_threshold,
    connect_topic_groups, digest, fit_topics_and_freeze, simulate_recent,
    split_development, topic_development, verify_freeze, view)
from gmail_classifier.text_features import clean_text


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_normalization_is_shared_and_pickle_stable(tmp_path):
    assert clean_text('AÇÃO https://example.test/123 user@example.test 44') == 'acao urltoken emailtoken numbertoken'
    assert 'hidden' not in clean_text('<p>Hello</p><script>hidden</script>')
    model = candidate('hybrid', thematic=True)
    texts = ['invoice payment', 'invoice bank', 'job interview', 'job employee', 'server error', 'server offline']
    model.fit(texts, ['Financeiro', 'Financeiro', 'RH', 'RH', 'Sistemas', 'Sistemas'])
    path = tmp_path / 'candidate.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.named_steps['tfidf'].transformer_list[0][1].preprocessor.__module__ == 'gmail_classifier.text_features'
    assert loaded.predict(['invoice bank']).tolist() == model.predict(['invoice bank']).tolist()


def test_group_split_never_moves_rare_signature_out_of_training():
    rows = [{'id':str(i), 'group_id':str(i//2), 'origin':'real', 'is_spam':bool(i % 2)} for i in range(16)]
    rows += [{'id':'rare', 'group_id':'rare', 'origin':'rare-source', 'is_spam':True}]
    before = copy.deepcopy(rows)
    result = split_development(rows)
    assert rows == before and result == split_development(rows)
    assert next(r for r in result if r['id'] == 'rare')['split'] == 'train'
    for group in {r['group_id'] for r in result}:
        assert len({r['split'] for r in result if r['group_id'] == group}) == 1


def test_threshold_selection_rejects_false_positive_shortcut_and_abstaining_precision():
    rows = [{'origin':source, 'is_spam':spam} for source in ('a','b') for spam in ([False]*20+[True]*10)]
    probability = np.array(([.1]*19+[.6]+[.95]*10)*2)
    protocol = {'spam_threshold_grid':[.5,.7,.9,1.0], 'min_validation_spam_suggestions':10,
                'min_validation_spam_precision':.98,'max_validation_spam_fpr_per_source':.01}
    policy = choose_spam_threshold(rows, probability, protocol)
    assert policy['feasible'] and policy['threshold'] >= .7
    assert all(m['false_positives'] == 0 for m in policy['by_source'].values())
    assert not choose_spam_threshold(rows, np.zeros(len(rows)), protocol)['feasible']


def test_topic_templates_and_declared_threads_are_connected():
    rows = [{'subject':'Invoice', 'body':'Pay at https://one.test/123', 'group_id':'a'},
            {'subject':'Invoice', 'body':'Pay at https://two.test/456', 'group_id':'b'},
            {'subject':'Different', 'body':'A different followup', 'group_id':'a'}]
    grouped = connect_topic_groups(rows)
    assert len({r['group_id'] for r in grouped}) == 1
    assert rows[0]['group_id'] == 'a'


def test_reserved_topic_content_never_enters_development(tmp_path):
    rows = [{'id':str(i), 'group_id':str(i), 'subject':word, 'body':word+' details',
             'is_spam':False, 'source':'email_reviewed', 'split':'train'}
            for i,word in enumerate(['alphabank','betabank','gammahire','deltahire','sealedtopic'])]
    write(tmp_path/'queue.json', {'records':rows})
    write(tmp_path/'protocol.json', {'topic_test_indices':[4]})
    write(tmp_path/'old-ai-categories.json', [])
    labels = {str(i):{'category':'Financeiro' if i<2 else 'RH','confidence':.9} for i in range(4)}
    write(tmp_path/'annotations.json', {'labels':labels})
    dev, coverage = topic_development(tmp_path)
    assert len(dev) == 4 and coverage['reserved'] == 1
    assert not any('sealedtopic' in r['body'] for r in dev)
    assert {r['split'] for r in dev} == {'train','validation'}


def test_annotation_provenance_and_test_access_require_freezing(tmp_path):
    write(tmp_path/'queue.json', {'records':[{'subject':'one','body':'body'},{'subject':'two','body':'sealed'}]})
    write(tmp_path/'protocol.json', {'topic_test_indices':[1], 'queue_sha256':digest(tmp_path/'queue.json')})
    write(tmp_path/'annotations.json', {'source':'email_ai_labeled','human_reviewed':False,'labels':{}})
    with pytest.raises(ValueError, match='Freeze'):
        view(tmp_path,0,10,test=True)
    annotation = {'index':0,'category':'RH','confidence':.9,'reason':'Employee recruitment'}
    annotate(tmp_path,[annotation])
    assert json.loads((tmp_path/'annotations.json').read_text())['human_reviewed'] is False
    with pytest.raises(ValueError, match='Development'):
        annotate(tmp_path,[dict(annotation,index=1)])
    with pytest.raises(ValueError, match='exists'):
        annotate(tmp_path,[dict(annotation,category='Financeiro')])


def test_topic_training_requires_both_private_and_ai_optins(tmp_path):
    with pytest.raises(ValueError, match='allow-private'):
        fit_topics_and_freeze(tmp_path)
    with pytest.raises(ValueError, match='allow-ai-labels'):
        fit_topics_and_freeze(tmp_path,allow_private=True)


def test_freeze_detects_model_or_input_mutation(tmp_path):
    files = {'model_sha256':'messages-real-v2.joblib','protocol_sha256':'protocol.json',
             'category_development_sha256':'category-development.json','queue_sha256':'queue.json',
             'annotations_development_sha256':'annotations-development-frozen.json'}
    for name in files.values():
        (tmp_path/name).write_bytes(b'original')
    write(tmp_path/'freeze.json', {key:digest(tmp_path/name) for key,name in files.items()})
    verify_freeze(tmp_path)
    (tmp_path/'messages-real-v2.joblib').write_bytes(b'modified')
    with pytest.raises(ValueError, match='frozen'):
        verify_freeze(tmp_path)


def test_failed_validation_cannot_emit_accepted_spam_or_topic(model, monkeypatch):
    model.provenance = {'spam_suggestions_enabled':False,'category_suggestions_enabled':False}
    monkeypatch.setattr(model.spam,'predict_proba',lambda x:np.array([[0.,1.]]))
    assert model.predict('Fatura','Boleto')['label'] == 'IA/Revisar'
    monkeypatch.setattr(model.spam,'predict_proba',lambda x:np.array([[1.,0.]]))
    assert model.predict('Fatura','Boleto')['label'] == 'IA/Revisar'


def test_recent_simulation_never_fabricates_accuracy_or_saves_text(model,tmp_path,monkeypatch):
    experiment = tmp_path/'experiment'
    experiment.mkdir()
    collection = tmp_path/'fresh.json'
    write(collection,{'source':'gmail_reviewed','records':[
        {'id':'a','subject':'Fatura','body':'boleto'}, {'id':'b','subject':'Unique','body':'unseen fresh message'}]})
    write(experiment/'development.json',{'records':[{'subject':'Fatura','body':'boleto','split':'train'}]})
    write(experiment/'category-development.json',{'records':[]})
    write(experiment/'final-evaluation.json',{})
    monkeypatch.setattr('gmail_classifier.experiment_v2.verify_freeze',lambda d:{'model_sha256':'frozen'})
    monkeypatch.setattr('gmail_classifier.experiment_v2.load_model',lambda p:model)
    monkeypatch.setattr('gmail_classifier.experiment_v2.private_home',lambda:tmp_path/'home')
    summary = simulate_recent(experiment,collection)
    assert summary['accuracy'] is None and summary['human_labels_available'] is False
    assert summary['mailbox_changes'] == 0 and summary['detected_training_overlap_count'] == 1
    saved = json.loads((experiment/'recent-predictions.json').read_text())
    assert all('subject' not in r and 'body' not in r for r in saved['predictions'])


def test_active_model_hash_and_review_gate(tmp_path,monkeypatch):
    from gmail_classifier.active_model import active_configuration, require_review
    monkeypatch.setattr('gmail_classifier.active_model.private_home',lambda:tmp_path)
    artifact = tmp_path/'model.joblib'
    artifact.write_bytes(b'trusted-own-model')
    write(tmp_path/'active-model.json',{'model':str(artifact),'report':str(tmp_path/'report.json'),
                                     'model_sha256':digest(artifact),'review_only':True})
    config = active_configuration()
    original = {'label':'IA/SPAM','action':'suggest_label','spam_probability':.99}
    reviewed = require_review(original,config)
    assert reviewed['label']=='IA/Revisar' and reviewed['proposed_label']=='IA/SPAM'
    assert original['label']=='IA/SPAM'
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        active_configuration()


def test_active_cli_keeps_experimental_hypotheses_in_review(model,tmp_path,monkeypatch,capsys):
    from gmail_classifier.cli import main
    monkeypatch.setattr('gmail_classifier.active_model.active_configuration',lambda:{'model':tmp_path/'model.joblib','review_only':True})
    monkeypatch.setattr('gmail_classifier.cli.load_model',lambda p:model)
    monkeypatch.setattr(model.spam,'predict_proba',lambda x:np.array([[.01,.99]]))
    main(['classify','--active','--text','Fatura boleto'])
    result = json.loads(capsys.readouterr().out)
    assert result['action']=='review' and result['proposed_label']=='IA/SPAM'
    assert result['category_candidate_condition']=='if_legitimate'
    assert result['category_candidate'] in model.category.classes_


def test_human_evaluator_requires_reviews_and_handles_absent_spam(model,tmp_path,monkeypatch):
    from gmail_classifier.experiment_v2 import evaluate_human
    from gmail_classifier.baseline import save_model
    save_model(model,tmp_path/'messages-real-v2.joblib')
    monkeypatch.setattr('gmail_classifier.experiment_v2.verify_freeze',lambda d:{})
    monkeypatch.setattr('gmail_classifier.experiment_v2.load_model',lambda p:model)
    monkeypatch.setattr('gmail_classifier.experiment_v2.known_training_keys',lambda d:set())
    collection = tmp_path/'review.json'
    data={'source':'gmail_reviewed','records':[{'id':'a','group_id':'a','subject':'Fatura','body':'Boleto',
        'is_spam':False,'category':'Financeiro','source':'gmail_reviewed',
        'human_reviewed':False,'privacy_reviewed':False,'exclude':False}]}
    write(collection,data)
    with pytest.raises(ValueError,match='pending'):
        evaluate_human(tmp_path,collection,tmp_path/'report.json','Reviewer')
    assert not (tmp_path/'report.json').exists()
    data['records'][0].update(human_reviewed=True,privacy_reviewed=True)
    write(collection,data)
    evaluate_human(tmp_path,collection,tmp_path/'report.json','Reviewer')
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['spam_recall_assessable'] is False
    assert report['spam_at_0_5']['spam_recall'] is None
    assert report['without_detected_training_overlap']['count']==1


def test_final_evaluation_cannot_be_repeated(tmp_path,monkeypatch):
    from gmail_classifier.experiment_v2 import evaluate
    monkeypatch.setattr('gmail_classifier.experiment_v2.verify_freeze',lambda d:{})
    write(tmp_path/'evaluation-started.json',{'one_time':True})
    with pytest.raises(ValueError,match='already started'):
        evaluate(tmp_path)


def test_preparation_refuses_untraceable_topic_quarantine(public_dataset,tmp_path,monkeypatch):
    from gmail_classifier.experiment_v2 import prepare
    home = tmp_path/'home'
    monkeypatch.setattr('gmail_classifier.experiment_v2.private_home',lambda:home)
    write(home/'email-category-review.json',{'records':[{'subject':f'missing {i}','body':'untraceable source'} for i in range(180)]})
    write(home/'dataset-ai-v1.json',[])
    public = tmp_path/'public.json'
    write(public,public_dataset)
    with pytest.raises(ValueError,match='Cannot trace'):
        prepare(tmp_path/'new-experiment',public)
    assert not (tmp_path/'new-experiment').exists()
