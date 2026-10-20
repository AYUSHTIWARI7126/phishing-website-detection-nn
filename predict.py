"""
Check URLs with the trained MLP.

usage:
    python predict.py http://paypal.com.verify-account.tk/login https://github.com
    python predict.py            # interactive mode
"""
import sys

import numpy as np

from features import extract_features
from nn_models import MLP

model, scaler, _ = MLP.load("results/mlp_model.npz")


def check(url):
    x = scaler.transform(np.array([extract_features(url)], dtype=float))
    p = model.predict_proba(x)[0]
    verdict = "PHISHING  " if p >= 0.5 else "LEGITIMATE"
    print(f"[{verdict}] P(phishing)={p:.3f}  {url}")


if len(sys.argv) > 1:
    for u in sys.argv[1:]:
        check(u)
else:
    print("Enter a URL (blank line to quit)")
    while (u := input("> ").strip()):
        check(u)
