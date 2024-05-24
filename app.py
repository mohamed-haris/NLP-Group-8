from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from transformers import pipeline
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
    os.remove(local_zip_path)

def replace_in_keys_and_values(data, replacements):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            new_key = key
            for old, new in replacements.items():
                new_key = new_key.replace(old, new)
            new_data[new_key] = replace_in_keys_and_values(value, replacements)
        return new_data
    elif isinstance(data, list):
        return [replace_in_keys_and_values(item, replacements) for item in data]
    elif isinstance(data, str):
        new_value = data
        for old, new in replacements.items():
            new_value = new_value.replace(old, new)
        return new_value
    else:
        return data


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

classifier_pipeline = pipeline(
    "token-classification",
    model= os.path.join(os.getcwd(), local_model_dir),
    tokenizer=tokenizer,
    aggregation_strategy='max'
)

app = Flask(__name__)    

# # Setup structured logging
logger = logging.getLogger('BERT_NER_Predictions')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Check if the handler is already added to avoid duplicates
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(handler)

for log_name in ('werkzeug', 'flask'):
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    # Remove all existing handlers
    log.handlers = []
    # Add our handler to the logger
    log.addHandler(handler)



@app.route('/')
def index():
    return "Hello from Group 8 NLP team!"

@app.route('/predict',methods = ['POST'])
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

    config_path = os.path.join(local_model_dir, 'config.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
    
    replacements = {
            'LABEL_0': 'LABEL-B-O',
            'LABEL_1': 'LABEL-B-AC',
            'LABEL_2': 'LABEL-B-LF',
            'LABEL_3': 'LABEL-I-LF',
    }
    
    updated_data = replace_in_keys_and_values(data, replacements)

    with open(config_path, 'w') as file:
        json.dump(updated_data, file, indent=4)
    
    print("JSON file has been updated with new dictionary names and values.")