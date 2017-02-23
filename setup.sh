virtualenv env
. env/bin/activate
pip install -r requirements.txt
python -m spacy.en.download
export KERAS_BACKEND="theano"
