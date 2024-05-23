from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim #
import logging
import json

MODEL_OF_CHOICE = "BERT"


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#models = AutoModelForTokenClassification.from_pretrained("D:\\NLP\\group\\bert_new_model_1.pth")
classifier_pipeline = pipeline(
    "token-classification",
    model="D:\\NLP\\group\\bert_new_model_1.pth",
    tokenizer=tokenizer,
    
    aggregation_strategy='max'
)
#classifier_pipeline = load('text_classification.joblib')
app = Flask(__name__)    

# Setup structured logging
logger = logging.getLogger('BERT_NER_Predictions')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@app.route('/predict',methods = ['POST'])
def predict():
    #methods = ['POST']
    request_data = request.get_json()
    text = request_data.get('text','')
    
    # Log the received text
    logger.info(json.dumps({'event': 'received_text', 'text': text}))
    #return jsonify(text)
    print(text)
    
    
    
    with torch.no_grad():
        outputs = classifier_pipeline(text)
        json_out = json.dumps(str(outputs))
        print(json_out) 
        logger.info(json.dumps({'event': 'outputs', 'json_out': json_out}))
      
    return (json_out)

if __name__ == '__main__':
    app.run(debug=True)