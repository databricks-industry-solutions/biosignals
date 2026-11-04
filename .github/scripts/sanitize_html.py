#!/usr/bin/env python3
"""HTML allowlist for the Pages site (.github/workflows/publish.yaml, convert_notebooks.py).

Every fragment rendered from repository content (Markdown, notebook cells, nbconvert output)
goes through ``clean`` before it is written into the page, so content that reaches the
default branch cannot run script on the site. ``clean`` uses nh3 (Python bindings to the
ammonia sanitizer) with an explicit allowlist:

* tags: text, headings, lists, links, images, code, tables, quotes and plain layout
  containers. ``script``, ``style``, ``iframe``, ``object``, ``embed``, ``svg``, ``math`` and
  every other tag are not allowed; script and style elements are removed with their content.
* attributes: ``id`` and ``class`` on any tag, ``href`` / ``title`` / ``data-section`` /
  ``data-anchor`` on links, ``src`` / ``alt`` / ``title`` / ``width`` / ``height`` on images,
  ``align`` on table cells and ``start`` on ordered lists. No ``on*`` event handler and no
  ``style`` attribute is allowed.
* URLs in ``href`` / ``src``: only the http, https and mailto schemes (compared
  case-insensitively, so ``JaVaScRiPt:`` is removed too), plus in-page ``#`` anchors; any
  other relative URL is removed. The page generator first rewrites links between rendered
  Markdown files to ``#`` anchors and every other relative link to the file on GitHub.
"""

import nh3

TAGS = {
    "a", "abbr", "b", "blockquote", "br", "code", "dd", "del", "details", "div", "dl", "dt",
    "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "kbd", "li", "ol", "p", "pre",
    "s", "samp", "span", "strong", "sub", "summary", "sup", "table", "tbody", "td", "tfoot",
    "th", "thead", "tr", "u", "ul",
}
# Removed together with their content (must not overlap TAGS). Any other tag that is not in
# TAGS is removed and its text kept.
CLEAN_CONTENT_TAGS = {"script", "style", "iframe", "object", "embed", "noscript", "template", "svg", "math"}
ATTRIBUTES = {
    "*": {"id", "class"},
    "a": {"href", "title", "data-section", "data-anchor"},
    "img": {"src", "alt", "title", "width", "height"},
    "td": {"align"},
    "th": {"align"},
    "ol": {"start"},
}
URL_SCHEMES = {"http", "https", "mailto"}


def _relative_url(url):
    """Keep in-page anchors; remove every other relative URL."""
    return url if url.startswith("#") else None


def clean(html):
    """``html`` with everything outside the allowlist removed."""
    return nh3.clean(
        html,
        tags=TAGS,
        clean_content_tags=CLEAN_CONTENT_TAGS,
        attributes=ATTRIBUTES,
        url_schemes=URL_SCHEMES,
        url_relative=_relative_url,
        link_rel="noopener noreferrer",
        strip_comments=True,
    )
