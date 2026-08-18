# Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score Loss

This repository provides an interactive Streamlit implementation of the amortized bandwidth selector developed for one-dimensional kernel density estimation (KDE).

The deployed tool takes a one-dimensional sample, constructs five summary features—sample size, mean, sample standard deviation, skewness, and kurtosis—and uses the final Gaussian-mixture-trained neural selector (GMM, K = 32) to predict a KDE bandwidth directly. The resulting truncated Gaussian KDE is evaluated on a finite support and can be downloaded as a CSV file.

## Online demo

After deployment, replace the placeholder below with the public Streamlit URL:

`https://<your-app-name>.streamlit.app`

## Repository structure

```text
amortized-kde/
├── app.py
├── inference.py
├── gmm32_selector.pt
├── requirements.txt
├── README.md
├── .gitignore
└── example_data/
    └── example_multicolumn_sample.csv
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

## Support options

The web app supports four finite-support choices:

1. Known support `[A, B]` supplied by the user.
2. Observed sample range treated as 90% of the total support width.
3. Observed sample range treated as 95% of the total support width.
4. Observed sample range treated as 99% of the total support width.

The 90%, 95%, and 99% options are deterministic support-extension rules. They are **not** confidence intervals and do **not** state that the observed sample range contains that percentage of probability mass.

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
- effective finite support;
- the estimated truncated Gaussian KDE;
- optional technical details including the five selector features and a numerical integral check.

The estimated density grid can also be downloaded as CSV with columns `x` and `density`.

## Paper

**Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score Loss**

Citation information can be added here once the final bibliographic record is available.
