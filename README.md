# Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score Loss

This repository provides an interactive Streamlit implementation of the amortized bandwidth selector developed for one-dimensional kernel density estimation (KDE).

The deployed tool takes a one-dimensional sample, constructs five summary features—sample size, mean, sample standard deviation, skewness, and kurtosis—and uses the final Gaussian-mixture-trained neural selector (GMM, K = 32) to predict a KDE bandwidth directly. The resulting truncated Gaussian KDE is evaluated on a finite support and can be downloaded as a CSV file.

## Online demo

`https://amortized-kde.streamlit.app`

## Repository structure

```text
app.py
inference.py
gmm32_selector.pt
requirements.txt
README.md
example_multicolumn_sample.csv
```

## Method

For a sample of size n, the selector uses the five raw features

```text
[n, mean, sample standard deviation, skewness, kurtosis]
```

The neural network has architecture `5 -> 128 -> 128 -> 1`, with ReLU hidden activations. Its scalar output is mapped to a positive bandwidth using `softplus`.

The public implementation loads the EMA weights from the final GMM K = 32 checkpoint. The selector was trained for sample sizes

```text
5 <= n <= 256
```

and on the reference support `[-1, 1]`. For a user-specified finite support `[A, B]`, the data are affinely mapped to `[-1, 1]`, the bandwidth is predicted on that scale, and the bandwidth and density are transformed back to the original scale.

## Interval specification

The web app provides two interval choices.

### 1. Known finite support

If the true finite support `[A, B]` is known, it can be supplied directly.

### 2. Automatic sample-adaptive working interval

Otherwise, let

\[
R=x_{\max}-x_{\min}.
\]

The app constructs

\[
A=x_{\min}-\frac{R}{N-1},
\qquad
B=x_{\max}+\frac{R}{N-1}.
\]

Equivalently,

\[
B-A=\frac{N+1}{N-1}R.
\]

This is a **uniform-reference, sample-size-adaptive working-interval rule**. Under a `Uniform(A,B)` reference model,

\[
E(R)=\frac{N-1}{N+1}(B-A),
\]

which motivates the expansion. The rule is affine equivariant.

The automatic interval is **not** a confidence interval and should not be interpreted as a general estimator of the mathematical support. If scientifically meaningful finite support bounds are known, the known-support option should be preferred.

## Input formats

Samples can be entered manually or uploaded as CSV/TXT.

For CSV files, all numeric cells are flattened row-by-row into a single one-dimensional sample. This means one-column, one-row, and multi-row/multi-column numeric layouts are all accepted. Text headers and empty cells are ignored. Any numeric metadata in the file will also be interpreted as sample values, so uploaded CSV files should contain only the intended sample values as numeric cells.

## Run locally

From the repository directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit, typically `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select the repository.
4. Set the entrypoint file to `app.py`.
5. Deploy the app.
6. After deployment, replace the placeholder URL in this README with the public `streamlit.app` URL.

The checkpoint file must remain in the repository root because `app.py` loads it using a path relative to the app file.

## Output

The app displays:

- sample size;
- predicted bandwidth on the original data scale;
- effective finite interval;
- the estimated truncated Gaussian KDE;
- optional technical details including the five selector features and a numerical integral check.

The estimated density grid can also be downloaded as CSV with columns `x` and `density`.

## Paper

**Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score Loss**

Citation information can be added here once the final bibliographic record is available.
