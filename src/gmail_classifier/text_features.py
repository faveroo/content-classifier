"""Shared, pickle-stable content normalization for training and inference."""

import re
import unicodedata

from .mail_input import HTMLText


def clean_text(text):
    text = text[:12000]
    if re.search(r"<(?:html|body|div|p)(?:\s|>)", text, re.I):
        parser = HTMLText()
        parser.feed(text)
        text = " ".join(parser.parts)
    text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[\w.+-]+\s*@\s*[\w.-]+(?:\s*\.\s*[\w-]+)+", " emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    return " ".join(text.split())
