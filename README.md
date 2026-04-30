# BioBERT Clinical NER — Flask Web App

Named Entity Recognition for clinical text using BioBERT-large fine-tuned on the NCBI Disease Corpus.

**Test F1: 0.8811 | Model: [dmis-lab/biobert-large-cased-v1.1](https://huggingface.co/sigebhanuprakash/biobert-ncbi-disease-ner)**

---

## Overview

This project fine-tunes BioBERT-large on the NCBI Disease Corpus to extract disease entity spans from biomedical text. The full pipeline covers model training, evaluation, deployment to Hugging Face Hub, and a live Flask web application hosted on Heroku.

**Development environment:** PyCharm (local) + Google Colab (training)

---

## Project Structure

```
biobert-ner-app/
├── app.py                  # Flask backend + inference logic
├── templates/
│   └── index.html          # Web UI
├── requirements.txt        # All dependencies
├── Procfile                # Heroku process declaration
└── .gitignore
```

---

## Workflow

This project was built in three stages:

1. **Training (Google Colab)** — Model was trained and evaluated on the NCBI Disease Corpus. The best checkpoint was selected based on validation F1.
2. **Model hosting (Hugging Face Hub)** — The fine-tuned model was pushed to Hugging Face: [sigebhanuprakash/biobert-ncbi-disease-ner](https://huggingface.co/sigebhanuprakash/biobert-ncbi-disease-ner)
3. **Deployment (Heroku via GitHub)** — The Flask app loads the model from Hugging Face at startup and is served via Heroku.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/biobert-ner-app.git
cd biobert-ner-app
```

### 2. Create a virtual environment

```bash
# Create and activate (Windows)
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> The app loads the model directly from Hugging Face Hub on first run — no manual model download needed.

---

## Deploy to Heroku

> **Note:** BioBERT-large is ~1.5 GB. The model is hosted on Hugging Face Hub and loaded at runtime to stay within Heroku's 500 MB slug limit.

```bash
# 1. Login to Heroku
heroku login

# 2. Create the Heroku app
heroku create biobert-ner-app
```

Then connect via GitHub:
1. Push this repo to GitHub
2. Go to Heroku Dashboard → your app → **Deploy** tab
3. Connect your GitHub repository
4. Click **Deploy Branch**

---

## API

**`POST /predict`**

Request:
```json
{ "text": "The patient has Huntington disease." }
```

Response:
```json
{
  "input": "The patient has Huntington disease.",
  "entities": [
    {
      "text": "huntington disease",
      "label": "Disease",
      "confidence": 0.9998,
      "low_conf": false
    }
  ]
}
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | dmis-lab/biobert-large-cased-v1.1 |
| Dataset | NCBI Disease Corpus |
| Labels | O, B-Disease, I-Disease |
| Hyperparameter search | Optuna TPE (10 trials) |
| Best learning rate | 3e-05 |
| Best batch size | 16 |
| Validation F1 | 0.8726 |
| **Test F1** | **0.8811** |
| Test Precision | 0.8601 |
| Test Recall | 0.9031 |

---

## Links

- Hugging Face Model: https://huggingface.co/sigebhanuprakash/biobert-ncbi-disease-ner
- Live App: https://biobert-ner-app.herokuapp.com
