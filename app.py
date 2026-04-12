import os
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────
# MODEL_DIR          = os.path.join(os.path.dirname(__file__), "model")
MODEL_DIR = "sigebhanuprakash/biobert-ncbi-disease-ner"
MAX_LENGTH         = 256
CONFIDENCE_THRESHOLD = 0.80

# Label map (NCBI Disease corpus: 3 labels)
id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}

# ── Load model & tokenizer once at startup ────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForTokenClassification.from_pretrained(
    MODEL_DIR,
    num_labels = len(id2label),
    id2label   = id2label,
    label2id   = label2id,
)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


# ── Inference helpers ─────────────────────────────────────────────
def predict_ner(text):
    """
    Run NER inference on raw biomedical text.
    Returns list of dicts: {"word": str, "label": str, "confidence": float}
    Subwords are merged — each entry is one complete word.
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    word_ids_list = encoding.word_ids()
    encoding.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**inputs).logits           # [1, seq_len, num_labels]

    probs       = F.softmax(logits[0], dim=-1).cpu()
    pred_ids    = probs.argmax(dim=-1).numpy()
    confidences = probs.max(dim=-1).values.numpy()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].cpu().numpy()
    )

    word_results  = []
    prev_word_id  = None

    for tok, wid, pid, conf in zip(tokens, word_ids_list, pred_ids, confidences):
        if wid is None:
            continue
        label = id2label[pid]
        if wid != prev_word_id:
            word_results.append({
                "word"       : tok,
                "label"      : label,
                "confidence" : float(conf),
            })
        else:
            suffix = tok[2:] if tok.startswith("##") else tok
            word_results[-1]["word"] += suffix
        prev_word_id = wid

    return word_results


def extract_entities(word_results):
    """
    Convert word-level BIO tags into entity spans.
    Returns list of {"text": str, "label": str, "confidence": float}
    """
    entities      = []
    current_words = []
    current_label = None
    current_confs = []

    for item in word_results:
        word  = item["word"]
        label = item["label"]
        conf  = item["confidence"]

        if label.startswith("B-"):
            if current_words:
                entities.append({
                    "text"       : " ".join(current_words),
                    "label"      : current_label,
                    "confidence" : round(sum(current_confs) / len(current_confs), 4),
                })
            current_words = [word]
            current_label = label[2:]
            current_confs = [conf]

        elif label.startswith("I-") and current_words:
            current_words.append(word)
            current_confs.append(conf)

        else:
            if current_words:
                entities.append({
                    "text"       : " ".join(current_words),
                    "label"      : current_label,
                    "confidence" : round(sum(current_confs) / len(current_confs), 4),
                })
                current_words = []
                current_label = None
                current_confs = []

    if current_words:
        entities.append({
            "text"       : " ".join(current_words),
            "label"      : current_label,
            "confidence" : round(sum(current_confs) / len(current_confs), 4),
        })

    return entities


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    word_results = predict_ner(text)
    entities     = extract_entities(word_results)

    # Add low-confidence flag
    for ent in entities:
        ent["low_conf"] = ent["confidence"] < CONFIDENCE_THRESHOLD

    return jsonify({
        "input"    : text,
        "entities" : entities,
        "words"    : word_results,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
