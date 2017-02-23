print __name__


from flask import Flask
import json
import datetime
import spacy
from flask import request
import pickle
from pathlib import Path
from keras.models import load_model
from sentiment.process_text import query_to_doc_array
import spacy


#set paths for data. 
#I should parse this from command line or config
data_path = Path('sentiment/trained_models')
model_path = data_path / 'sentiment_model_og'
info_path = data_path / 'sentiment_model_og_info.pkl'

#load trained model
model_info = pickle.load(open(str(info_path)))
token_2_id, id_2_token = model_info['token_2_id'], model_info['id_2_token']
model = load_model(str(model_path))
print "LSTM model loaded"

#responsible for text -> vector
nlp = spacy.load('en', parse = False, entity = False)
expected_length = model.layers[0].input_length
print "Text processor loaded"


app = Flask(__name__)

@app.route('/')
def go_somewhere_else():
    return 'check out static/sentiment.html!'


@app.route('/sentiment')
def get_sentiment():
    text = unicode(request.args.get('query'))
    vector = query_to_doc_array(text, token_2_id, nlp, expected_length)
    prediction = model.predict(vector)
    return str({'sentiment':prediction})


if __name__ == "__main__":
    app.run()
