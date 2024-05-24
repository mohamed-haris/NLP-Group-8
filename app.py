from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging
import os
import json
import gdown
import zipfile

MODEL_OF_CHOICE = "BERT"
file_id = '1Fa1CdPZTrhwjSPEZJTBQN_BI4SW3uHG9'
model_url = f'https://drive.google.com/uc?id={file_id}'
local_zip_path = 'bert_model.pth.zip'
local_model_dir = 'bert_model.pth'

if not os.path.exists(local_model_dir):
    gdown.download(model_url, local_zip_path, quiet=False)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(local_model_dir)
classifier_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy='max'
)

app = Flask(__name__)

# Setup structured logging
logger = logging.getLogger('BERT_NER_Predictions')
logger.setLevel(logging.WARNING)
handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    text = request_data.get('text', '')

    logger.warning(json.dumps({'event': 'received_text', 'text': text}))

    with torch.no_grad():
        outputs = classifier_pipeline(text)
        output = {}
        for i, item in enumerate(outputs):
            del item['score']
            del item['start']
            del item['end']
            output[f"word-{i}"] = item

        json_out = json.dumps(output)
        logger.warning(json.dumps({'event': 'outputs', 'json_out': json_out}))

    return jsonify(output)

if __name__ == '__main__':
    app.run()