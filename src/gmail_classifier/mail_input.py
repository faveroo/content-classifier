"""Provider-independent, offline mail parsing. Never renders HTML or fetches URLs."""

from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import mailbox
from pathlib import Path
import re

MAX_BYTES = 2_000_000
MAX_TEXT = 20_000


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}:
            self.hidden += 1
        elif tag in {"br", "p", "div", "li", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"} and self.hidden:
            self.hidden -= 1
        self.parts.append(" ")

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


def _body(part):
    if part.get_content_disposition() == "attachment" or part.get_filename():
        return ""
    if part.get_content_type() == "message/rfc822":
        return ""
    if part.is_multipart():
        parts = list(part.iter_parts())
        if part.get_content_subtype() == "alternative":
            plain = [p for p in parts if p.get_content_type() == "text/plain"]
            for child in plain or parts:
                if text := _body(child):
                    return text
            return ""
        return "\n".join(filter(None, (_body(child) for child in parts)))
    if part.get_content_type() not in {"text/plain", "text/html"}:
        return ""
    raw = part.get_payload(decode=True) or b""
    try:
        text = raw.decode(part.get_content_charset() or "utf-8", errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    if part.get_content_type() == "text/html":
        parser = HTMLText()
        parser.feed(text)
        text = " ".join(parser.parts)
    return text


def parse_email(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Message exceeds the 2 MB parsing limit")
    message = BytesParser(policy=policy.default).parsebytes(raw)
    # Only subject and visible body become features; filter verdict headers are ignored.
    subject = str(message.get("Subject", "")).strip()[:MAX_TEXT]
    body = _body(message).strip()[:MAX_TEXT]
    if not (subject + body).strip():
        raise ValueError("Message has no usable text")
    return {"subject": subject, "body": body}


def iter_messages(path):
    """Read .eml, .mbox, a directory of .eml, or a UTF-8 plain-text message."""
    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*.eml")):
            yield from iter_messages(child)
    elif path.suffix.lower() == ".mbox":
        box = mailbox.mbox(path, factory=None, create=False)
        try:
            for key in box.iterkeys():
                yield parse_email(box.get_bytes(key))
        finally:
            box.close()
    elif path.suffix.lower() == ".eml":
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Message exceeds the 2 MB parsing limit")
        yield parse_email(path.read_bytes())
    elif path.suffix.lower() == ".txt":
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Message exceeds the 2 MB parsing limit")
        body = path.read_text(encoding="utf-8")[:MAX_TEXT]
        if not body.strip():
            raise ValueError("Message has no usable text")
        yield {"subject": "", "body": body}
    else:
        raise ValueError("Use .eml, .mbox, .txt or a directory of .eml files")


def canonical_text(subject, body):
    return " ".join((subject + " " + body).casefold().split())


def template_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text.casefold())
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[\w-]+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    return " ".join(re.findall(r"\w+", text))


def thread_subject(subject):
    return re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", subject.casefold()).strip()


def content_group_keys(row):
    text = canonical_text(row["subject"], row["body"])
    keys = {("exact", text), ("template", template_text(text))}
    body = template_text(row["body"])
    if len(body) >= 80:
        keys.add(("body", body))
    subject = thread_subject(row["subject"])
    if len(subject) >= 12 and len(subject.split()) >= 3:
        keys.add(("subject", subject))
    return keys
