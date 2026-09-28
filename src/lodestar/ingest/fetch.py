"""Fetch a page's HTML."""

import logging

import httpx

log = logging.getLogger(__name__)

# Some sites refuse requests that don't look like a browser.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def fetch_page(url: str, timeout: float = 20.0) -> str:
    """Return the HTML at `url`, following redirects. Raises httpx.HTTPError on failure."""
    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        follow_redirects=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    if str(resp.url) != url:
        log.info("redirected: %s -> %s", url, resp.url)
    log.info("fetched %s (%d bytes)", url, len(resp.text))
    return resp.text
