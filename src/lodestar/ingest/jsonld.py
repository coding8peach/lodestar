"""Find a schema.org JobPosting in a page's JSON-LD blocks."""

import json
import logging
from typing import Any

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def _is_job_posting(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    t = node.get("@type")
    return t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t)


def _walk(node: Any):
    """Yield candidate nodes: the node itself, list items, and @graph members."""
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
    elif isinstance(node, dict):
        yield node
        if "@graph" in node:
            yield from _walk(node["@graph"])


def find_job_posting(html: str) -> dict | None:
    """Return the first JSON-LD JobPosting object in the page, or None."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        text = script.string or script.get_text()
        if not text or not text.strip():
            continue
        try:
            data = json.loads(text, strict=False)  # strict=False: some sites leave raw newlines in strings
        except json.JSONDecodeError:
            log.warning("skipping malformed JSON-LD block")
            continue
        for node in _walk(data):
            if _is_job_posting(node):
                return node
    return None
