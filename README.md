# Phishing Website Detection using Neural Networks
*Neural Computation & Applications – Mini Project*

## 1. Problem
Phishing websites copy real sites (banks, PayPal, Amazon…) to steal passwords.
The URL usually gives them away: IP addresses instead of domain names, `@` symbols,
brand names hidden in sub-domains (`paypal.com.verify-account.tk`), cheap TLDs,
URL shorteners and so on.
**Goal:** given a URL, a neural network should classify it as **phishing (1)** or **legitimate (0)**.

## 2. Pipeline
```
URL ──► Feature extraction (20 features) ──► z-score normalisation ──► Neural network ──► P(phishing)
```

### Features (input layer, `features.py`)
| Lexical | Domain / host | Security / tricks |
|---|---|---|
| URL length, path length | host length, no. of dots | uses HTTPS |
| no. of hyphens, digits, digit ratio | no. of sub-domains | `@` symbol, `//` redirect |
| special characters, query params | host entropy (randomness) | IP address as host, non-standard port |
| suspicious words (login, verify…) | suspicious TLD (.tk, .xyz…) | URL shortener, brand name not in real domain |

## 3. NCA topics used (all implemented from scratch in NumPy – `nn_models.py`)

| # | Model / concept | What it shows |
|---|---|---|
| 1 | **Perceptron** (Rosenblatt) | step activation, learning rule `w ← w + η(t − y)x`; fails to converge on non-linearly separable data |
| 2 | **ADALINE** | linear neuron, **delta rule / LMS**, gradient descent on mean-squared error |
| 3 | **Multi-Layer Perceptron** | forward propagation, **back-propagation** (chain rule), sigmoid/tanh/ReLU activations, binary cross-entropy loss, mini-batch SGD, **momentum**, **L2 weight decay**, **early stopping**, Xavier/He initialisation |
| 4 | **RBF Network** | Gaussian radial basis functions, centres found by **k-means** (unsupervised), output weights by regularised least squares (pseudo-inverse) |
| 5 | **Kohonen Self-Organising Map** | **unsupervised competitive learning**, BMU, neighbourhood function, decaying learning rate & radius, U-matrix; neurons labelled by majority vote to use it as a classifier |
| – | Experiments | effect of **activation function** and **number of hidden neurons**; comparison with scikit-learn MLP |
| – | Evaluation | train/validation/test split, accuracy, precision, recall, F1, confusion matrix, permutation feature importance |

Key back-prop equations used in the code:
```
Forward :  z(l) = a(l-1)·W(l) + b(l),   a(l) = f(z(l)),   ŷ = σ(z(L))
Loss    :  L = −[t·log ŷ + (1−t)·log(1−ŷ)]
Output  :  δ(L) = ŷ − t                    (sigmoid + cross-entropy)
Hidden  :  δ(l) = (δ(l+1)·W(l+1)ᵀ) ⊙ f′(z(l))
Gradient:  ∂L/∂W(l) = a(l-1)ᵀ·δ(l) / m + λW(l)
Update  :  v ← βv − η∂L/∂W,   W ← W + v      (momentum)
```

## 4. How to run
```bash
pip install -r requirements.txt
python dataset.py        # (optional) just generate & preview the dataset
python train.py          # trains all models, prints results, saves plots in results/
python predict.py http://paypal.com.verify-account.tk/login https://github.com
python predict.py        # interactive mode
```
Using a real dataset instead (e.g. UCI *Phishing Websites* dataset or Kaggle
*Web page Phishing Detection*, exported as CSV):
```bash
python train.py --csv phishing.csv --label-col Result
```
(`predict.py` only works with the URL-feature model trained on the default dataset.)

## 5. Results (synthetic dataset, 4000 URLs, 3 % label noise, 600 test samples)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Perceptron | 0.702 | 0.675 | 0.824 | 0.742 |
| ADALINE | 0.952 | 0.967 | 0.939 | 0.953 |
| **MLP (32-16, ReLU)** | **0.978** | 0.972 | **0.987** | **0.979** |
| RBF network | 0.927 | 0.953 | 0.904 | 0.928 |
| Kohonen SOM | 0.963 | 0.948 | 0.984 | 0.965 |
| sklearn MLP (reference) | 0.960 | 0.974 | 0.949 | 0.961 |

Activation experiment: sigmoid 0.978, tanh 0.977, ReLU 0.978 (ReLU/tanh converge in fewer epochs).
Hidden neurons: 2 → 0.957, 4 → 0.972, 8 → 0.978, 16 → 0.975, 64 → 0.977 (beyond ~8 neurons no gain).

Plots in `results/`: learning curves, activation comparison, model comparison,
confusion matrix, SOM U-matrix / phishing map, feature importance.

## 6. Observations (useful for the report / viva)
* **Perceptron is worst** – the data is not linearly separable, so the Rosenblatt rule keeps
  oscillating and never converges (see its error curve). This motivates multi-layer networks.
* **ADALINE** does much better than the perceptron with the *same* linear decision boundary,
  because it minimises a smooth error (MSE) by gradient descent instead of reacting only to mistakes.
* **MLP** gives the best results: hidden layers learn non-linear combinations of features
  (e.g. "brand name present AND domain is not the brand AND suspicious TLD").
* High **recall** matters most here – a missed phishing site (false negative) is more
  dangerous than a false alarm.
* **Early stopping** on validation loss prevents over-fitting (training stopped at ~65 epochs).
* The **SOM**, trained *without labels*, still separates phishing and legitimate URLs into
  different regions of the map – the two classes form natural clusters in feature space.
* Accuracy is capped around 97 % by the intentional 3 % label noise.

## 7. Limitations / future work
* Dataset is synthetic (generated from known phishing patterns); results on real data
  will be lower – try the UCI/Kaggle datasets with `--csv`.
* Only URL features are used; page content (HTML forms, external links), WHOIS domain age
  and SSL certificate info would improve detection.
* Character-level CNN/LSTM on the raw URL could learn features automatically.

## Files
| File | Purpose |
|---|---|
| `features.py` | URL → 20 numeric features |
| `dataset.py` | synthetic dataset generator + CSV loader |
| `nn_models.py` | Perceptron, ADALINE, MLP, RBF, SOM (from scratch) |
| `train.py` | training, experiments, metrics, plots |
| `predict.py` | classify new URLs with the saved MLP |
