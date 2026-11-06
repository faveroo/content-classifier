"""Best-effort local minimization; human privacy review is still required."""

import re


def redact(text: str) -> str:
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", " [URL] ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[\w-]+", " [EMAIL] ", text)
    text = re.sub(r"(?<!\w)\+?\d[\d\s()./-]{5,}\d(?!\w)", " [NUMERO] ", text)
    return " ".join(text.split())[:20000]
