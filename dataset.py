"""
Dataset builder.

Option A (default): generates a labelled synthetic URL dataset that mimics
common phishing tricks (IP hosts, '@' symbol, brand names in sub-domains,
look-alike hyphenated domains, URL shorteners, cheap TLDs ...).
Some noise and overlap is added on purpose so the problem is not trivial.

Option B: load a real CSV dataset (e.g. the UCI "Phishing Websites" dataset
or Kaggle "Web page Phishing Detection") with --csv.
"""
import csv
import random
import string

import numpy as np

from features import extract_features, FEATURE_NAMES

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "amazon.in", "amazon.com",
    "paypal.com", "microsoft.com", "apple.com", "stackoverflow.com",
    "youtube.com", "linkedin.com", "netflix.com", "bbc.co.uk", "nytimes.com",
    "reddit.com", "python.org", "numpy.org", "kaggle.com", "irctc.co.in",
    "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "flipkart.com",
    "zomato.com", "swiggy.com", "coursera.org", "nptel.ac.in", "iitb.ac.in",
    "medium.com", "quora.com", "instagram.com", "facebook.com", "x.com",
]
LEGIT_PATHS = [
    "", "/", "/about", "/search", "/wiki/Neural_network", "/products/item",
    "/user/profile", "/docs/stable/index.html", "/news/world", "/help",
    "/courses/ml", "/watch", "/questions/12345/how-to", "/blog/2024/post",
    "/login", "/account/settings", "/cart", "/en-us/store",
]
BRANDS = ["paypal", "apple", "amazon", "microsoft", "google", "facebook",
          "netflix", "instagram", "chase", "hdfc", "sbi", "icici"]
BAD_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "info", "ru", "buzz",
            "live", "click", "com", "net", "org"]
BAD_WORDS = ["login", "verify", "secure", "account", "update", "confirm",
             "signin", "billing", "unlock", "wallet", "support", "alert"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "is.gd", "cutt.ly", "rb.gy"]


def _rand_str(n, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(n))


def legit_url():
    scheme = "https" if random.random() < 0.9 else "http"
    dom = random.choice(LEGIT_DOMAINS)
    sub = random.choice(["", "", "www.", "www.", "m.", "en.", "docs.", "mail."])
    path = random.choice(LEGIT_PATHS)
    if random.random() < 0.3:
        params = "&".join(f"{_rand_str(random.randint(1, 5), string.ascii_lowercase)}="
                          f"{_rand_str(random.randint(1, 8))}"
                          for _ in range(random.randint(1, 4)))
        path += "?" + params
    return f"{scheme}://{sub}{dom}{path}"


def phishing_url():
    brand = random.choice(BRANDS)
    word = random.choice(BAD_WORDS)
    tld = random.choice(BAD_TLDS)
    scheme = "https" if random.random() < 0.45 else "http"
    trick = random.randint(0, 6)
    if trick == 0:   # IP address host
        host = ".".join(str(random.randint(1, 254)) for _ in range(4))
        url = f"{scheme}://{host}/{brand}/{word}.php"
    elif trick == 1:  # brand in subdomain
        url = f"{scheme}://{brand}.com.{word}-{_rand_str(5)}.{tld}/{word}"
    elif trick == 2:  # '@' trick
        url = f"{scheme}://{brand}.com@{_rand_str(8)}.{tld}/{word}"
    elif trick == 3:  # hyphenated look-alike
        url = f"{scheme}://{brand}-{word}-{_rand_str(4)}.{tld}/index.html"
    elif trick == 4:  # shortener
        url = f"https://{random.choice(SHORTENERS)}/{_rand_str(7, string.ascii_letters + string.digits)}"
    elif trick == 5:  # long random host + redirect
        url = (f"{scheme}://{_rand_str(random.randint(10, 25))}.{tld}"
               f"/{word}//redirect?url=http://{brand}.com&id={_rand_str(12)}")
    else:            # subtle typo-squat, looks almost normal
        typo = brand.replace("o", "0").replace("l", "1").replace("a", "4", 1)
        url = f"{scheme}://www.{typo}.com/{word}"
    if trick != 4 and random.random() < 0.15:  # non-standard port
        sch, rest = url.split("://", 1)
        host, _, tail = rest.partition("/")
        url = f"{sch}://{host}:{random.choice([8080, 8443, 2096])}/{tail}"
    return url


def generate_synthetic(n=4000, label_noise=0.03, seed=42):
    random.seed(seed)
    urls, labels = [], []
    for i in range(n):
        y = i % 2  # balanced: 1 = phishing, 0 = legitimate
        urls.append(phishing_url() if y else legit_url())
        labels.append(y)
    # label noise: real datasets are never perfectly labelled
    for i in range(n):
        if random.random() < label_noise:
            labels[i] = 1 - labels[i]
    X = np.array([extract_features(u) for u in urls], dtype=float)
    y = np.array(labels, dtype=float)
    return X, y, urls


def save_csv(path, X, y, urls):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url"] + FEATURE_NAMES + ["label"])
        for u, row, lab in zip(urls, X, y):
            w.writerow([u] + list(row) + [int(lab)])


def load_csv(path, label_col=None):
    """Load any numeric CSV. Non-numeric columns (e.g. url) are skipped.
    Labels -1/1 (UCI style) or 'phishing'/'legitimate' are mapped to 1/0."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    label_idx = header.index(label_col) if label_col else len(header) - 1

    def to_label(v):
        v = v.strip().lower()
        if v in ("phishing", "1", "bad", "malicious"):
            return 1.0
        if v in ("legitimate", "0", "-1", "good", "benign"):
            return 0.0
        raise ValueError(f"Unknown label value: {v}")

    feat_idx, names = [], []
    for j, name in enumerate(header):
        if j == label_idx:
            continue
        try:
            float(data[0][j])
            feat_idx.append(j)
            names.append(name)
        except ValueError:
            pass
    X = np.array([[float(r[j]) for j in feat_idx] for r in data])
    y = np.array([to_label(r[label_idx]) for r in data])
    return X, y, names


if __name__ == "__main__":
    X, y, urls = generate_synthetic()
    save_csv("phishing_dataset.csv", X, y, urls)
    print(f"Saved {len(y)} samples ({int(y.sum())} phishing) to phishing_dataset.csv")
    for u in urls[:6]:
        print("  ", u)
