"""Read-only Gmail transport. No training logic or mailbox write operations."""

import base64
import json
import re
import secrets
from email.header import decode_header, make_header
from html.parser import HTMLParser
from pathlib import Path

from .private_storage import private_path, write_private_json
from .text_privacy import redact


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = [READONLY_SCOPE]
MAX_MESSAGES = 500
MAX_TEXT = 20000


class GmailError(ValueError):
    """Safe user-facing message, never an upstream response or private content."""


def _google_dependencies():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise GmailError('Install Gmail dependencies: python -m pip install -e ".[gmail]"') from None
    return Request, Credentials, InstalledAppFlow, build


def authenticate(credentials_path: Path, token_path: Path, *, interactive=False):
    """Only gmail-auth may open the consent flow; collection never does so implicitly."""
    credentials_path, token_path = private_path(credentials_path), private_path(token_path)
    if credentials_path == token_path:
        raise GmailError("Credentials and token must use different paths")
    Request, Credentials, InstalledAppFlow, _ = _google_dependencies()
    try:
        credentials = None
        if token_path.exists():
            data = json.loads(token_path.read_text(encoding="utf-8"))
            if set(data.get("scopes", [])) != set(SCOPES):
                raise GmailError("Token must grant only gmail.readonly; use a separate read-only OAuth client")
            credentials = Credentials.from_authorized_user_info(data, SCOPES)
            if credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                except Exception:
                    if not interactive:
                        raise GmailError("Read-only token refresh failed; run gmail-auth to renew browser consent") from None
                    # Only the explicit auth command may replace an expired/revoked grant.
                    # Keep the previous token file untouched until new scopes are verified.
                    credentials = None
        if credentials is None or not credentials.valid:
            if not interactive:
                raise GmailError("Authenticate first with gmail-auth")
            config = json.loads(credentials_path.read_text(encoding="utf-8"))
            if "installed" not in config:
                raise GmailError("Create a Desktop app OAuth client in Google Cloud")
            flow = InstalledAppFlow.from_client_config(config, SCOPES, autogenerate_code_verifier=True)
            credentials = flow.run_local_server(
                host="127.0.0.1", port=0, timeout_seconds=180,
                authorization_prompt_message="",
                success_message="Autorizacao concluida. Voce pode fechar esta janela.",
                prompt="consent", access_type="offline", include_granted_scopes="false",
            )
        granted = credentials.granted_scopes
        if granted is not None and set(granted) != set(SCOPES):
            raise GmailError("OAuth returned unexpected scopes; revoke this grant and use a dedicated client")
        write_private_json(token_path, json.loads(credentials.to_json()), overwrite=True)
        return credentials
    except GmailError:
        raise
    except Exception:
        # Google errors may contain account details, URLs, tokens or message text.
        raise GmailError("OAuth failed; check the client, test user, token and browser consent") from None


def connect(credentials_path: Path, token_path: Path):
    credentials = authenticate(credentials_path, token_path)
    _, _, _, build = _google_dependencies()
    try:
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)
    except Exception:
        raise GmailError("Unable to initialize Gmail; check API enablement and network") from None


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "tr"}:
            self.text.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif tag in {"p", "div", "li", "tr"}:
            self.text.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def _body_text(part: dict) -> str:
    headers = {h["name"].lower(): h["value"] for h in part.get("headers", [])}
    if part.get("filename") or headers.get("content-disposition", "").lower().startswith("attachment"):
        return ""
    mime = part.get("mimeType", "")
    if mime == "message/rfc822":
        return ""
    children = part.get("parts", [])
    if children:
        if mime == "multipart/alternative":
            plain = [p for p in children if p.get("mimeType") == "text/plain"]
            if plain:
                return "\n".join(filter(None, (_body_text(p) for p in plain)))
            return next((text for p in children if (text := _body_text(p))), "")
        return "\n".join(filter(None, (_body_text(p) for p in children)))
    data = part.get("body", {}).get("data")
    if mime not in {"text/plain", "text/html"} or not data:
        return ""  # Never fetch attachments, including text stored as attachmentId.
    charset = re.search(r'charset\s*=\s*["\']?([^\s;"\']+)', headers.get("content-type", ""), re.I)
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    try:
        text = raw.decode(charset.group(1) if charset else "utf-8", errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if mime == "text/html":
        parser = _HTMLText()
        parser.feed(text)
        text = " ".join(parser.text)
    return text


def collect(service, *, query: str, limit=50, include_spam=False) -> dict:
    if type(limit) is not int or not 1 <= limit <= MAX_MESSAGES:
        raise GmailError("Collection limit must be between 1 and 500")
    if not isinstance(query, str) or not query.strip():
        raise GmailError("An explicit Gmail search query is required")
    rows, groups, seen, page_tokens = [], {}, set(), set()
    page_token = None
    fetched = skipped = 0
    # Pseudonyms are local to this collection; no raw Gmail IDs or lookup map are saved.
    try:
        messages = service.users().messages()
        while fetched < limit:
            options = dict(userId="me", q=query, maxResults=min(limit - fetched, 100),
                           includeSpamTrash=include_spam)
            if page_token:
                options["pageToken"] = page_token
            page = messages.list(**options).execute()
            items = page.get("messages", [])
            for item in items:
                if fetched >= limit:
                    break
                fetched += 1
                if item["id"] in seen:
                    skipped += 1
                    continue
                seen.add(item["id"])
                message = messages.get(userId="me", id=item["id"], format="full").execute()
                payload = message.get("payload", {})
                subject = next((h["value"] for h in payload.get("headers", [])
                                if h["name"].lower() == "subject"), "")
                subject = redact(str(make_header(decode_header(subject))))
                body = redact(_body_text(payload))
                if not subject and not body:
                    skipped += 1
                    continue
                thread = message.get("threadId") or item.get("threadId") or item["id"]
                group_id = groups.setdefault(thread, "g-" + secrets.token_hex(12))
                rows.append(dict(id="m-" + secrets.token_hex(12), group_id=group_id,
                                 subject=subject, body=body, is_spam=None, category=None,
                                 source="gmail_reviewed", split=None, human_reviewed=False,
                                 privacy_reviewed=False, exclude=False))
            next_token = page.get("nextPageToken")
            if not next_token or not items:
                break
            if next_token in page_tokens:
                raise GmailError("Gmail returned a repeated page token; collection stopped")
            page_tokens.add(next_token)
            page_token = next_token
    except GmailError:
        raise
    except Exception:
        raise GmailError("Gmail collection failed; check access and network. No collection was saved") from None
    return {"format_version": 1, "source": "gmail_reviewed", "fetched": fetched,
            "skipped": skipped, "records": rows}
