#!/usr/bin/env python3
"""Fetch scholarly paper titles from URLs.

Usage: python3 scripts/fetch_paper_title.py <url> [url ...]
Output: one line per url as "<url>\\t<title>" (or "<url>\\tERROR: ...").

Strategy: arXiv links -> arXiv Atom API (cleanest); everything else -> HTML,
preferring the <meta name="citation_title"> tag that scholarly sites (arXiv,
IEEE, ACM, Springer, OpenReview, ...) expose, then og:title, then <title>.
Used by the paper-title skill (content/paperreading/papertitle.skill).
"""
import re
import ssl
import sys
import html as htmllib
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; paper-title-skill/1.0)"}
CTX = ssl.create_default_context()


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read().decode("utf-8", "replace")


def clean(s):
    return htmllib.unescape(re.sub(r"\s+", " ", s)).strip()


def arxiv_id(url):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url, re.I)
    return m.group(1) if m else None


def from_arxiv(aid):
    data = fetch(f"http://export.arxiv.org/api/query?id_list={aid}")
    m = re.search(r"<entry>.*?<title>(.*?)</title>", data, re.S)
    return clean(m.group(1)) if m else None


def from_html(url):
    # Normalize an arXiv PDF link to its abstract page so meta tags are present.
    url = re.sub(r"(arxiv\.org)/pdf/([\w.]+?)(?:\.pdf)?$", r"\1/abs/\2", url, flags=re.I)
    page = fetch(url)
    patterns = [
        r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_title["\']',
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
        r"<title[^>]*>([^<]+)</title>",
    ]
    for pat in patterns:
        m = re.search(pat, page, re.I | re.S)
        if m:
            return clean(m.group(1))
    return None


# Anti-bot / JS-wall interstitial titles that must not be mistaken for a paper.
BLOCKED = re.compile(
    r"verify(ing)? you|just a moment|attention required|access denied|"
    r"\bforbidden\b|are you a robot|captcha|enable javascript|请稍候|人机验证|域名拦截",
    re.I,
)


def title_for(url):
    aid = arxiv_id(url)
    if aid:
        try:
            t = from_arxiv(aid)
            if t:
                return t
        except Exception:
            pass  # fall through to HTML scraping
    t = from_html(url)
    if t and BLOCKED.search(t):
        return None  # interstitial page, not a real title -> caller should retry
    return t


def main():
    urls = [u for u in sys.argv[1:] if u.strip()]
    if not urls:
        print("usage: python3 scripts/fetch_paper_title.py <url> [url ...]", file=sys.stderr)
        sys.exit(2)
    ok = True
    for u in urls:
        try:
            t = title_for(u)
            print(f"{u}\t{t}" if t else f"{u}\tERROR: title not found")
            ok = ok and bool(t)
        except Exception as e:
            print(f"{u}\tERROR: {type(e).__name__}: {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
