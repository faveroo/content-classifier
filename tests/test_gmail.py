import base64
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gmail_classifier import gmail
from gmail_classifier.cli import main
from gmail_classifier.dataset import validate_records
from gmail_classifier.private_storage import CHECKOUT, private_path, write_private_json
from gmail_classifier.review import prepare_dataset, reviewed_records, simulate


def part(text, mime="text/plain", **kwargs):
    return {"mimeType": mime, "body": {"data": base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")}, **kwargs}


def message(identifier="raw-id", thread="raw-thread"):
    return {"id": identifier, "threadId": thread, "payload": {
        "mimeType": "multipart/alternative",
        "headers": [{"name": "Subject", "value": "Fatura de fake@example.test"},
                    {"name": "From", "value": "Nome Privado <fake@example.test>"}],
        "parts": [part("Pagamento em https://example.test/secret 123.456.789-00"),
                  part("<p>HTML duplicate</p>", "text/html")]}}


def service_for(pages, messages):
    endpoint = Mock(spec=["list", "get"])
    endpoint.list.return_value.execute.side_effect = pages
    endpoint.get.return_value.execute.side_effect = messages
    service = Mock()
    service.users.return_value.messages.return_value = endpoint
    return service, endpoint


def test_collection_is_readonly_bounded_and_minimized():
    service, endpoint = service_for(
        [{"messages": [{"id": "raw-a"}], "nextPageToken": "page2"},
         {"messages": [{"id": "raw-b"}, {"id": "never-fetched"}]}],
        [message("raw-a"), message("raw-b")])
    sample = gmail.collect(service, query="newer_than:30d", limit=2)
    assert sample["fetched"] == 2
    assert len(sample["records"]) == endpoint.get.call_count == 2
    first, second = sample["records"]
    assert first["group_id"] == second["group_id"]
    assert first["id"] != second["id"]
    assert first["is_spam"] is None and first["split"] is None
    assert first["human_reviewed"] is first["privacy_reviewed"] is False
    saved = json.dumps(sample)
    for secret in ("raw-a", "raw-b", "raw-thread", "fake@example.test", "https://example.test", "123.456", "Nome Privado", "HTML duplicate"):
        assert secret not in saved
    assert endpoint.list.call_args_list[0].kwargs == {
        "userId": "me", "q": "newer_than:30d", "maxResults": 2, "includeSpamTrash": False}
    assert endpoint.list.call_args_list[1].kwargs["pageToken"] == "page2"
    assert endpoint.get.call_args_list[0].kwargs == {"userId": "me", "id": "raw-a", "format": "full"}


def test_mime_ignores_attachments_and_active_html():
    payload = {"mimeType": "multipart/mixed", "parts": [
        part("<style>secret style</style><p>Corpo &amp; texto</p><script>secret script</script>", "text/html"),
        part("secret attachment", filename="private.txt"),
        part("secret attachment 2", headers=[{"name": "Content-Disposition", "value": "attachment"}]),
        {"mimeType": "message/rfc822", "parts": [part("secret forwarded attachment")]},
        {"mimeType": "text/plain", "body": {"attachmentId": "never-download"}},
    ]}
    text = gmail._body_text(payload)
    assert "Corpo & texto" in text
    assert "secret" not in text
    assert gmail.redact("oi fake@example.test +55 (11) 99999-1234 www.example.test/path") == "oi [EMAIL] [NUMERO] [URL]"


@pytest.mark.parametrize("limit", [0, 501, True, -1])
def test_collection_rejects_invalid_limit_before_network(limit):
    service = Mock()
    with pytest.raises(gmail.GmailError):
        gmail.collect(service, query="in:inbox", limit=limit)
    service.users.assert_not_called()


def test_collection_errors_hide_private_content():
    service, endpoint = service_for([], [])
    endpoint.list.side_effect = RuntimeError("private email fake@example.test token=secret")
    with pytest.raises(gmail.GmailError) as error:
        gmail.collect(service, query="in:inbox")
    assert "secret" not in str(error.value) and "fake@" not in str(error.value)


def test_empty_mailbox_and_spam_option():
    service, endpoint = service_for([{}], [])
    sample = gmail.collect(service, query="in:spam", include_spam=True)
    assert sample["records"] == []
    assert endpoint.list.call_args.kwargs["includeSpamTrash"] is True
    endpoint.get.assert_not_called()


def test_private_paths_and_exclusive_write(tmp_path):
    with pytest.raises(ValueError):
        private_path(CHECKOUT / "data/private.json")
    repo = tmp_path / "other-repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    with pytest.raises(ValueError):
        private_path(repo / "sub/private.json")
    output = tmp_path / "private.json"
    write_private_json(output, {"test": True})
    with pytest.raises(FileExistsError):
        write_private_json(output, {})
    assert json.loads(output.read_text()) == {"test": True}


def fake_auth(monkeypatch):
    credentials = SimpleNamespace(valid=True, expired=False, refresh_token="fake",
                                  granted_scopes=gmail.SCOPES,
                                  to_json=lambda: json.dumps({"scopes": gmail.SCOPES, "token": "fake"}),
                                  refresh=Mock())
    credential_type, flow_type = Mock(), Mock()
    credential_type.from_authorized_user_info.return_value = credentials
    flow_type.from_client_config.return_value.run_local_server.return_value = credentials
    monkeypatch.setattr(gmail, "_google_dependencies", lambda: (Mock(), credential_type, flow_type, Mock()))
    return credentials, credential_type, flow_type


def test_auth_only_requests_readonly_and_local_callback(tmp_path, monkeypatch):
    credentials, _, flow_type = fake_auth(monkeypatch)
    config, token = tmp_path / "credentials.json", tmp_path / "token.json"
    config.write_text('{"installed": {"client_id": "fake"}}')
    assert gmail.authenticate(config, token, interactive=True) is credentials
    assert flow_type.from_client_config.call_args.args[1] == gmail.SCOPES
    options = flow_type.from_client_config.return_value.run_local_server.call_args.kwargs
    assert options["host"] == "127.0.0.1" and options["port"] == 0
    assert options["include_granted_scopes"] == "false"
    assert options["authorization_prompt_message"] == ""
    assert json.loads(token.read_text())["scopes"] == gmail.SCOPES


def test_auth_rejects_broad_tokens_before_loading_credentials(tmp_path, monkeypatch):
    _, credential_type, flow_type = fake_auth(monkeypatch)
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"scopes": ["https://mail.google.com/"]}))
    with pytest.raises(gmail.GmailError, match="only gmail.readonly"):
        gmail.authenticate(tmp_path / "credentials.json", token)
    credential_type.from_authorized_user_info.assert_not_called()
    flow_type.from_client_config.assert_not_called()


def test_collection_auth_never_opens_browser_and_refreshes_readonly_token(tmp_path, monkeypatch):
    credentials, _, flow_type = fake_auth(monkeypatch)
    config, token = tmp_path / "credentials.json", tmp_path / "token.json"
    with pytest.raises(gmail.GmailError, match="Authenticate first"):
        gmail.authenticate(config, token)
    flow_type.from_client_config.assert_not_called()
    token.write_text(json.dumps({"scopes": gmail.SCOPES}))
    credentials.expired = True
    gmail.authenticate(config, token)
    credentials.refresh.assert_called_once()


def test_auth_rejects_unexpected_grant_and_sanitizes_failure(tmp_path, monkeypatch):
    credentials, _, flow_type = fake_auth(monkeypatch)
    config, token = tmp_path / "credentials.json", tmp_path / "token.json"
    config.write_text('{"installed": {}}')
    credentials.granted_scopes = gmail.SCOPES + ["https://mail.google.com/"]
    with pytest.raises(gmail.GmailError, match="unexpected scopes"):
        gmail.authenticate(config, token, interactive=True)
    assert not token.exists()
    flow_type.from_client_config.side_effect = RuntimeError("secret token")
    with pytest.raises(gmail.GmailError) as error:
        gmail.authenticate(config, token, interactive=True)
    assert "secret" not in str(error.value)


def test_expired_grant_needs_explicit_auth_and_preserves_token_on_failure(tmp_path, monkeypatch):
    credentials, _, flow_type = fake_auth(monkeypatch)
    config, token = tmp_path / "credentials.json", tmp_path / "token.json"
    config.write_text('{"installed": {}}')
    original = json.dumps({"scopes": gmail.SCOPES, "token": "old-fake"})
    token.write_text(original)
    credentials.expired = True
    credentials.refresh.side_effect = RuntimeError("secret OAuth response")
    with pytest.raises(gmail.GmailError, match="run gmail-auth") as error:
        gmail.authenticate(config, token)
    assert "secret" not in str(error.value)
    flow_type.from_client_config.assert_not_called()
    assert token.read_text() == original
    flow_type.from_client_config.return_value.run_local_server.side_effect = TimeoutError()
    with pytest.raises(gmail.GmailError):
        gmail.authenticate(config, token, interactive=True)
    assert token.read_text() == original
    flow_type.from_client_config.return_value.run_local_server.side_effect = None
    assert gmail.authenticate(config, token, interactive=True) is credentials
    assert json.loads(token.read_text())["token"] == "fake"


def review_collection(records):
    rows = copy.deepcopy(records)
    for row in rows:
        row.update(source="gmail_reviewed", human_reviewed=True, privacy_reviewed=True, exclude=False)
    return {"format_version": 1, "source": "gmail_reviewed", "records": rows}


def test_review_requires_both_reviews_labels_and_group_integrity(records):
    collection = review_collection(records)
    assert len(reviewed_records(collection)) == 80
    collection["records"][0]["human_reviewed"] = False
    with pytest.raises(ValueError, match="review"):
        reviewed_records(collection)
    collection["records"][0]["human_reviewed"] = True
    collection["records"][0]["privacy_reviewed"] = False
    with pytest.raises(ValueError, match="review"):
        reviewed_records(collection)
    collection["records"][0]["privacy_reviewed"] = True
    collection["records"][-1]["group_id"] = collection["records"][0]["group_id"]
    with pytest.raises(ValueError, match="boundary"):
        reviewed_records(collection)


def test_review_exclusion_and_preparation(records, tmp_path):
    collection = review_collection(records)
    collection["records"][0].update(exclude=True, human_reviewed=False, privacy_reviewed=False)
    source, output = tmp_path / "review.json", tmp_path / "dataset.json"
    source.write_text(json.dumps(collection))
    rows = prepare_dataset(source, output)
    assert len(rows) == 79
    assert "human_reviewed" not in rows[0]
    with pytest.raises(ValueError):
        validate_records(rows)  # Explicit opt-in is required even after review.
    assert validate_records(rows, allow_private=True) == rows


def test_private_training_cli_reports_actual_source_and_stays_offline(records, tmp_path, capsys):
    dataset, model, report = (tmp_path / filename for filename in ("data.json", "model.joblib", "report.json"))
    dataset.write_text(json.dumps(reviewed_records(review_collection(records))))
    main(["train", "--allow-private", "--dataset", str(dataset), "--model", str(model), "--report", str(report)])
    assert json.loads(report.read_text())["source"] == "gmail_reviewed"
    assert "Synthetic smoke test" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["train", "--allow-private", "--dataset", str(dataset)])
    assert error.value.code == 2  # Defaults point into the checkout and must be rejected.


def test_offline_simulation_never_copies_message_text(model, records, tmp_path):
    from gmail_classifier.baseline import save_model
    collection = review_collection(records)
    collection["records"] = collection["records"][:2]
    source, artifact, output = (tmp_path / filename for filename in ("review.json", "model.joblib", "predictions.json"))
    source.write_text(json.dumps(collection))
    save_model(model, artifact)
    assert len(simulate(source, artifact, output)) == 2
    saved = json.loads(output.read_text())
    assert saved["mailbox_changes"] == 0
    assert set(saved["predictions"][0]) == {"id", "spam_probability", "category", "category_confidence", "action", "label"}
    with pytest.raises(ValueError):
        simulate(source, artifact, output)


def test_cli_checks_output_before_connecting(tmp_path, monkeypatch, capsys):
    connect = Mock()
    monkeypatch.setattr(gmail, "connect", connect)
    output = tmp_path / "review.json"
    output.write_text("existing")
    with pytest.raises(SystemExit):
        main(["gmail-collect", "--query", "in:inbox", "--output", str(output)])
    connect.assert_not_called()
    assert output.read_text() == "existing"
    assert "existing" not in capsys.readouterr().err
