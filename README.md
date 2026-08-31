# Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score

This repository provides an interactive implementation of amortized bandwidth selection for one-dimensional kernel density estimation (KDE). Instead of applying a fixed rule or solving a new optimisation problem for every sample, a trained neural selector maps the sample directly to a Gaussian-KDE bandwidth.

The current Streamlit app accepts user-supplied data and uses the final Gaussian-mixture-trained selector (GMM, $K=32$) to construct a truncated-and-renormalised KDE on a finite interval. The project is being extended into an interactive comparison tool covering multiple bandwidth selectors, bounded and unbounded KDEs, and simulated distributions with known targets.

## Online demo

[https://amortized-kde.streamlit.app](https://amortized-kde.streamlit.app)

## Current implementation

The deployed app currently supports:

- manual sample entry and CSV/TXT upload;
- a known finite support or an automatic sample-adaptive working interval;
- bandwidth prediction using the final GMM $K=32$ amortized selector;
- a truncated-and-renormalised Gaussian KDE;
- display of the predicted bandwidth, effective interval and numerical integral;
- download of the estimated density grid as CSV.

## Planned interactive comparison

The next version will add two complementary modes.

### 1. User-data mode

For an uploaded or manually entered sample, users will be able to select one or more bandwidth methods:

- Amortized selector;
- Silverman's rule;
- Sheather–Jones selector;
- least-squares cross-validation (LSCV).

All selected KDEs will be drawn in the same figure using different colours. They will use the same sample, interval and evaluation grid; only the selected bandwidth will differ. The legend will report the bandwidth returned by each method.

Two density views will be available:

- **Bounded KDE:** the Gaussian KDE is truncated and renormalised on $[A,B]$.
- **Unbounded KDE:** the ordinary Gaussian KDE is defined on $\mathbb R$; only its displayed horizontal range is finite.

Because the underlying distribution of user-supplied data is unknown, this mode will not report a target-based logarithmic score. Its purpose is direct visual comparison of the resulting density estimates.

### 2. Simulation mode

Simulation mode will generate a fresh task in the browser from a known underlying distribution. It will contain three experiment types:

| Experiment type | Available target distributions | Amortized checkpoint |
| --- | --- | --- |
| Gaussian | Gaussian task generated under the Gaussian experiment setting | `gaussian_selector.pt` |
| Distribution family | Gaussian, Laplace, Student-$t$, Gamma, Beta, Logistic, Lognormal, Bimodal, Trimodal, Spike-and-slab, or Multi-family | `multifamily_selector.pt` |
| GMM $K=32$ | Fresh bounded Gaussian mixture generated under the paper's $K=32$ setting | `gmm32_selector.pt` |

In the **Multi-family** option, each task is drawn from one of the ten component families. It is not a new density obtained by mixing all ten families together.

For a chosen sample size and random seed, the app will:

1. generate a new observed sample;
2. display the true target density;
3. overlay the KDEs selected by the user;
4. generate an independent scoring sample from the same target;
5. report the empirical logarithmic score for each method.

For an independent scoring sample $Y_1,\ldots,Y_m$, the reported score will be

$$
\widehat L(\widehat f)
=-\frac{1}{m}\sum_{j=1}^{m}\log_2 \widehat f(Y_j),
$$

measured in bits. Smaller values are better.

Users will also be able to request tests for additional target distributions through a GitHub issue.

### 3. Reproducible model examples

Two compact training examples are planned:

- `train_gaussian.py`: Gaussian task generation, model training and inference;
- `train_gmm32.py`: fresh bounded GMM $K=32$ task generation, model training and inference.

These scripts will expose the essential training logic without including all paper figures, bootstrap calculations or large experimental notebooks.

### 4. Privacy-preserving usage statistics

The deployed app may record aggregate events such as page visits, approximate anonymous users who successfully generate a KDE, successful generation counts and run times. It will not store uploaded sample values, filenames, IP addresses or other directly identifying information.

## Checkpoints

The repository contains three trained selectors.

| File | Training tasks | Input | Network | Inference weights |
| --- | --- | --- | --- | --- |
| `gaussian_selector.pt` | Gaussian | sample size $n$ | $1\to4\to4\to1$ | `model_state_dict` |
| `multifamily_selector.pt` | Ten distribution families | five sample features | $5\to128\to128\to1$ | `ema_model` |
| `gmm32_selector.pt` | Fresh bounded GMM, $K=32$ | five sample features | $5\to128\to128\to1$ | `ema_model` |

The Multi-family and GMM selectors use the raw feature vector

```text
[n, mean, sample standard deviation, skewness, kurtosis]
```

All checkpoints use double-precision parameters. The trained sample-size range is

```text
5 <= n <= 256
```

Predictions outside this range are extrapolations and have not been validated in the reported experiments. For user-supplied data with no known family label, the GMM $K=32$ selector is the default amortized method.

## Method

### Affine transfer to a finite interval

The Multi-family and GMM selectors were trained on the reference interval $[-1,1]$. For a finite working interval $[A,B]$, let

$$
a=\frac{B-A}{2},
\qquad
z=\frac{x-(A+B)/2}{a}.
$$

The transformed observations lie on the reference scale. If the selector returns bandwidth $h_z$, the bandwidth on the original scale is

$$
h_x=a h_z.
$$

The density is transformed back using

$$
\widehat f_x(x)=\frac{1}{a}\widehat f_z(z).
$$

### Bounded KDE

For bandwidth $h>0$, the ordinary Gaussian KDE is

$$
\widetilde f_h(x)
=\frac{1}{nh}\sum_{i=1}^{n}
\phi\!\left(\frac{x-x_i}{h}\right).
$$

On a finite interval $[A,B]$, the displayed bounded estimator is

$$
\widehat f_{h,[A,B]}(x)
=\frac{\widetilde f_h(x)}
{\displaystyle \frac{1}{n}\sum_{i=1}^{n}
\left[
\Phi\!\left(\frac{B-x_i}{h}\right)
-\Phi\!\left(\frac{A-x_i}{h}\right)
\right]}
\mathbf 1\{A\le x\le B\}.
$$

Thus the estimator integrates to one over the selected finite interval.

## Interval specification

The web app provides two interval choices.

### 1. Known finite support

If scientifically meaningful finite bounds $[A,B]$ are known, they should be supplied directly.

### 2. Automatic sample-adaptive working interval

Otherwise, let

$$
R=x_{\max}-x_{\min}.
$$

For a sample of size $n$, the app constructs

$$
A=x_{\min}-\frac{R}{n-1},
\qquad
B=x_{\max}+\frac{R}{n-1}.
$$

Equivalently,

$$
B-A=\frac{n+1}{n-1}R.
$$

This is a uniform-reference, sample-size-adaptive working-interval rule. For $n$ independent observations from a uniform distribution on an interval of length $L$,

$$
E(R)=\frac{n-1}{n+1}L,
$$

which motivates the expansion. The rule is affine equivariant.

The automatic interval is **not** a confidence interval and is not a general estimator of the mathematical support. Known scientific bounds should be preferred when available.

## Input formats

Samples can be entered manually or uploaded as CSV/TXT.

For tabular files, all numeric cells are flattened row-by-row into a single one-dimensional sample. One-column, one-row and multi-row/multi-column numeric layouts are accepted. Text headers and empty cells are ignored. Numeric metadata will also be interpreted as observations, so uploaded files should contain only the intended sample values in numeric cells.

## Repository structure

Current files:

```text
Amortized-KDE/
├── app.py
├── inference.py
├── gaussian_selector.pt
├── multifamily_selector.pt
├── gmm32_selector.pt
├── example_multicolumn_sample.csv
├── requirements.txt
└── README.md
```

Planned modules:

```text
models.py                  # Neural-network definitions and checkpoint registry
kde_methods.py             # Amortized, Silverman, Sheather-Jones and LSCV methods
distributions.py           # Gaussian, ten-family and GMM task generators
evaluation.py              # Independent-sample logarithmic-score evaluation
plotting.py                # Shared colours, legends and density figures
analytics.py               # Anonymous aggregate usage events
train_gaussian.py          # Compact Gaussian training example
train_gmm32.py             # Compact GMM K=32 training example
tests/                     # Loading, positivity, integration and generator tests
```

## Run locally

From the repository directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit, typically [http://localhost:8501](http://localhost:8501).

## Scope and limitations

- The implementation is for one-dimensional numerical samples.
- The amortized selectors were trained for $5\le n\le256$.
- The bounded estimator depends on a user-specified or automatically constructed finite interval.
- The automatic interval is a working rule, not a confidence statement about the true support.
- Simulation-mode log scores are Monte Carlo estimates and therefore vary with the random seed and scoring-sample size.
- For user data without a known target distribution, density curves can be compared but a target-based log score is unavailable.

## Paper

Junyi Liang and Hailiang Du, **“Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score.”**

Preprint: [arXiv:2608.20445](https://arxiv.org/abs/2608.20445)

```bibtex
@article{liang2026amortized,
  title         = {Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score},
  author        = {Liang, Junyi and Du, Hailiang},
  year          = {2026},
  eprint        = {2608.20445},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```
