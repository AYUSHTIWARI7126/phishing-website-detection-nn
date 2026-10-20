"""
Feature extraction: turns a raw URL string into a numeric feature vector
that can be fed to a neural network (the "input layer").
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure", "account", "update",
    "confirm", "banking", "bank", "password", "webscr", "ebayisapi", "suspend",
    "unlock", "wallet", "billing", "support", "alert", "free", "bonus", "lucky",
]
SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "click",
                   "country", "work", "info", "ru", "cn", "buzz", "live"}
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
              "buff.ly", "cutt.ly", "rb.gy", "tiny.cc", "shorturl.at"}
BRANDS = ["paypal", "apple", "amazon", "microsoft", "google", "facebook",
          "netflix", "instagram", "chase", "wellsfargo", "hdfc", "sbi", "icici"]

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots_host", "num_hyphens",
    "num_subdomains", "num_digits", "digit_ratio", "num_query_params",
    "num_special_chars", "host_entropy", "has_at_symbol", "has_ip_host",
    "uses_https", "has_double_slash_redirect", "has_port", "suspicious_tld",
    "is_shortener", "suspicious_word_count", "brand_not_in_domain",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def extract_features(url: str) -> list:
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    lower = url.lower()

    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    # registered domain ~ last two labels (good enough for this project)
    reg_domain = ".".join(parts[-2:]) if len(parts) >= 2 else host

    num_digits = sum(ch.isdigit() for ch in url)
    brand_in_url = any(b in lower for b in BRANDS)
    brand_in_domain = any(b in reg_domain for b in BRANDS)

    return [
        len(url),
        len(host),
        len(path),
        host.count("."),
        url.count("-"),
        max(len(parts) - 2, 0),
        num_digits,
        num_digits / max(len(url), 1),
        len([q for q in query.split("&") if q]),
        sum(url.count(c) for c in "~_%=&?!*$+"),
        _entropy(host),
        int("@" in url),
        int(bool(IP_RE.match(host))),
        int(parsed.scheme == "https"),
        int("//" in path),
        int(parsed.port is not None),
        int(tld in SUSPICIOUS_TLDS),
        int(host in SHORTENERS),
        sum(w in lower for w in SUSPICIOUS_WORDS),
        int(brand_in_url and not brand_in_domain),
    ]
