# Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score

This repository provides an interactive implementation of amortized bandwidth selection for one-dimensional kernel density estimation (KDE). Instead of applying a fixed rule or solving a new optimisation problem for every sample, a trained neural selector maps the sample directly to a Gaussian-KDE bandwidth.

The Streamlit app compares amortized and classical bandwidth selectors on user-supplied data and displays the repeated-experiment benchmark figures reported by the project.

## Online demo

[https://amortized-kde.streamlit.app](https://amortized-kde.streamlit.app)

## Application workflows

### Data

The Data workflow supports:

- manual sample entry and CSV/TXT upload;
- a known finite support or an automatic sample-adaptive working interval;
- automatic checkpoint selection from the Gaussian, Multi-family and GMM $K=32$ selectors;
- Silverman's rule, Sheather–Jones and LSCV comparisons;
- bounded and ordinary unbounded Gaussian KDE views;
- download of the estimated density grids as CSV.

Available bandwidth methods are:

- Amortized selector;
- Silverman's rule;
- Sheather–Jones selector;
- least-squares cross-validation (LSCV).

All selected KDEs are drawn using the same sample and method-specific bandwidths.

Two density views are available:

- **Bounded KDE:** the Gaussian KDE is truncated and renormalised on $[A,B]$.
- **Unbounded KDE:** the ordinary Gaussian KDE is defined on $\mathbb R$; only its displayed horizontal range is finite.

Because the full target density and its parameters are not available from an observed sample, this workflow does not report a target-based logarithmic score.

### Simulation

The Simulation workflow displays only aggregate repeated-experiment figures: the Gaussian benchmark, the ten-family aggregate and family-specific results, and the bounded GMM $K=32$ benchmark. It does not generate or score a single random task in the browser, because one realization is not representative of aggregate method performance.

Users may request an additional family-specific benchmark through a GitHub issue. The request does not include uploaded sample data.

## Reproducible training scripts

The repository includes two standalone training scripts:

- `training/train_gaussian.py`: Gaussian task generation, $n$-only bandwidth-ratio training and best-checkpoint selection;
- `training/train_gmm32.py`: exact bounded GMM $K=32$ generation, five-feature training, EMA validation and best-checkpoint selection.

Both files are self-contained and do not depend on variables defined in a notebook. Their default settings reproduce the reported training protocols. A small `--quick` mode is provided only to verify that the code executes; it does not reproduce the trained models.

Training outputs are written under `training_outputs/`, so running either script does not overwrite the published checkpoints in `models/`.

## Privacy-preserving usage statistics

The deployed app records page visits and successful Data-workflow KDE generations using an anonymous per-session identifier. It does not store uploaded sample values, filenames, IP addresses or distribution-request free text.

## Checkpoints

The repository contains three trained selectors.

| File | Training tasks | Input | Network | Inference weights |
| --- | --- | --- | --- | --- |
| `models/gaussian_selector.pt` | Gaussian | sample size $n$ | $1\to4\to4\to1$ | `model_state_dict` |
| `models/multifamily_selector.pt` | Ten distribution families | five sample features | $5\to128\to128\to1$ | `ema_model` |
| `models/gmm32_selector.pt` | Fresh bounded GMM, $K=32$ | five sample features | $5\to128\to128\to1$ | `ema_model` |

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

The deployment entry point and configuration files stay at the repository
root. Application modules, checkpoints, figures and training scripts are kept
in separate directories:

```text
Amortized-KDE/
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
├── kde_app/
│   ├── __init__.py
│   ├── analytics.py
│   ├── config.py
│   ├── core.py
│   ├── classical_bandwidth_selectors.py
│   ├── neural_bandwidth_selectors.py
│   ├── kde_estimators.py
│   ├── kde_plotting.py
│   ├── ui_components.py
│   ├── data_page.py
│   └── simulation_page.py
├── models/
│   ├── gaussian_selector.pt
│   ├── multifamily_selector.pt
│   └── gmm32_selector.pt
├── figures/
│   └── *.png
├── training/
│   ├── train_gaussian.py
│   └── train_gmm32.py
└── .streamlit/
    └── secrets.toml.example
```

## Run locally

### Web app

From the repository directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit, typically [http://localhost:8501](http://localhost:8501).

### Training scripts

Run a small execution check:

```bash
python3 training/train_gaussian.py --quick
python3 training/train_gmm32.py --quick
```

Run the complete default training protocols:

```bash
python3 training/train_gaussian.py
python3 training/train_gmm32.py
```

The complete GMM training is computationally expensive: its maximum configuration contains 40,000 optimisation steps with 256 fresh GMM tasks per step.

## Scope and limitations

- The implementation is for one-dimensional numerical samples.
- The amortized selectors were trained for $5\le n\le256$.
- The bounded estimator depends on a user-specified or automatically constructed finite interval.
- The automatic interval is a working rule, not a confidence statement about the true support.
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
