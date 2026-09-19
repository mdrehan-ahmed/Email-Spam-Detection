# Email Spam Detection

A Streamlit web app that checks whether an email is likely to be **Spam** or **Authentic** using Machine Learning.

## Features

- Paste email text and analyze it.

- Upload `.txt`, `.eml`, `.png`, `.jpg`, or `.jpeg` files.

- Extract text from screenshots using EasyOCR.

- Analyze multiple files together.

- View spam probability, authenticity score, and confidence.

- See batch summary, charts, and download a CSV report.

- Responsive dark interface for laptop, tablet, and mobile.

## Project Structure

```
project/
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

## Dataset

Place the dataset here:

```
data/spam.csv
```

The commonly used dataset has these columns:

```
v1 = ham or spam
v2 = message text
```

The application converts `ham` to `0` and `spam` to `1` for model training.

## Run Locally

Open the project folder in VS Code and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

## Note

This model is trained mainly on short spam/ham messages. Results may vary for long emails, HTML emails, attachments, or unfamiliar writing styles. The prediction is an estimate and should not be treated as a guaranteed security decision.