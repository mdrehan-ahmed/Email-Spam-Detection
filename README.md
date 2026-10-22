# Email Spam Detection
Live app: [https://rehan-email-spam-detection.streamlit.app/]

A responsive Streamlit web application that analyzes pasted emails, uploaded files, and screenshots to estimate whether a message is likely **Spam**, **Authentic**, or **Uncertain**.

## Features

- Paste email text for analysis.

- Upload `.txt`, `.eml`, `.png`, `.jpg`, and `.jpeg` files.

- Extract text from screenshots with EasyOCR.

- Analyze one or multiple emails/files together.

- Show spam probability, authenticity score, and confidence.

- Use three result categories: Likely Spam, Likely Authentic, and Uncertain.

- Display batch summary, charts, results table, and downloadable CSV report.

- Responsive dark interface for laptop, tablet, and mobile.

- Cache the machine-learning model and OCR reader for faster repeated use.

## Project Structure

```
Email-Spam-Detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── spam.csv
```

## Technologies

- Python

- Streamlit

- Pandas and NumPy

- Scikit-learn

- EasyOCR

- Pillow

- Vega-Lite charts through Streamlit

## Dataset Requirement

The current `app.py` loads the training dataset from this exact path:

```
data/spam.csv
```

The file should be a CSV dataset with one of these supported column formats:

```
v1, v2
```

or:

```
Category, Message
```

The usual labels are `ham` and `spam`. The application converts them internally to `0` and `1` for model training.

If `data/spam.csv` is missing or invalid, the app uses a small built-in fallback dataset, which is only for demonstration.

## Run Locally

From the project folder, run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit, normally:

```
http://localhost:8501
```

## Push Updates to GitHub

After changing files in VS Code:

```bash
git status
git add .
git commit -m "Improve spam classification and update app configuration"
git push
```

Because Streamlit Community Cloud is connected to the GitHub repository, a successful push to the deployed branch triggers a new deployment automatically.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.

2. Open [Streamlit Community Cloud](https://share.streamlit.io/).

3. Select the connected app.

4. Set the repository branch to `main` and the main file to `app.py`.

5. Deploy or reboot the app after a new commit.

## Result Interpretation

```
0%–40%   → Likely Authentic
40%–60%  → Uncertain / Needs Review
60%–100% → Likely Spam
```

The prediction is based mainly on email text patterns. It does not independently verify sender identity, links, SPF, DKIM, DMARC, or email headers. The model is mainly trained on short spam/ham messages, so long policy emails and unfamiliar writing styles may produce uncertain results.