# BioBERT Clinical NER — Flask Web App

Named Entity Recognition for clinical text using BioBERT-large fine-tuned on the NCBI Disease Corpus.

**Test F1: 0.8811 | Model: dmis-lab/biobert-large-cased-v1.1**

---

## Project Structure

```
biobert-ner-app/
├── app.py                  # Flask backend + inference logic
├── templates/
│   └── index.html          # Web UI
├── model/                  # Place your saved BioBERT files here
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── requirements.txt
├── Procfile
└── .gitignore
```

---

## Step 1 — Add your model files

Download your saved model from Google Drive (`optuna-trial/final-model/`) and place all files into the `model/` folder.

---

## Step 2 — Run locally

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open http://localhost:5000 in your browser.

---

## Step 3 — Deploy to Heroku

> **Note:** BioBERT-large is ~1.5 GB. Heroku's slug limit is 500 MB.
> Host the model on Hugging Face Hub and load it at startup, OR use a Heroku dyno with Git LFS.

```bash
# Login to Heroku
heroku login

# Create app
heroku create biobert-ner-app

# Push to Heroku via GitHub (recommended)
# 1. Push this repo to GitHub
# 2. Connect repo in Heroku Dashboard → Deploy tab
# 3. Click "Deploy Branch"
```

---

## API

**POST /predict**

Request:
```json
{ "text": "The patient has Huntington disease." }
```

Response:
```json
{
  "input": "The patient has Huntington disease.",
  "entities": [
    { "text": "huntington disease", "label": "Disease", "confidence": 0.9998, "low_conf": false }
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
| Optimizer | Optuna (10 trials) |
| Best lr | 2e-05 |
| Best batch size | 16 |
| Validation F1 | 0.8726 |
| Test F1 | 0.8811 |
| Test Precision | 0.8601 |
| Test Recall | 0.9031 |
