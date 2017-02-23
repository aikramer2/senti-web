import os
os.environ['KERAS_BACKEND'] = 'theano'

import nltk
import spacy
import process_text

from functools import partial
import numpy as np
from collections import Counter
import pickle

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout

def flatten(array):
	"""
	takes an iterable of iterables, flattens into single list

	Parameters
	----------
		
		array: iterable of iterables


	Returns
	---------
		flattened_array (list)
	"""
	return [element for subarray in array for element in subarray]


def collect_corpus(corpus_id):
	"""
	collects an nltk corpus

	Parameters
	----------
		
		corpus_id(string): nltk corpus name


	Returns
	---------
		docs (list), labels (list)

	"""	
	nltk.download(corpus_id)
	corpus = getattr(nltk.corpus, corpus_id)
	fileids = corpus.fileids()
	labels =  map(lambda ID: corpus.categories(ID), fileids)
	docs = map(lambda ID: corpus.raw(ID), fileids)
	return docs, labels

def process_docs(docs):
	"""
	processes each document according to the "process_text" function.

	Parameters
	----------
		
		docs(list): documents to process


	Returns
	---------
		processed_docs(list)

	"""		
	processor = partial(process_text.process, **{'nlp':nlp})
	processed_docs = map(processor, docs)
	return processed_docs


def docs_to_seq_matrix(processed_docs, max_words = 5000):
	"""
	converts a list of lists of words to a matrix of dim (ndocs, max_seq_len)
	Values correspond to word ids

	Parameters
	----------
		
		processed_docs(list): list of documents post-processing
		max_words(int): the maximum number of words to include in the vocabulary


	Returns
	---------
		doc_matrix (np.ndarray), token_2_id(dict), id_2_token(dict)

	"""		
	words = flatten(processed_docs)
	word_counts = Counter()
	word_counts.update(words)
	common_words = map(lambda pair: pair[0], word_counts.most_common(max_words))
	clean_doc = lambda doc: filter(lambda word: word in common_words,doc)
	processed_docs = map(clean_doc, processed_docs)
	token_2_id = {j:i for i,j in enumerate(common_words)}
	id_2_token = {i:j for i,j in enumerate(common_words)}
	doc_2_id = lambda doc: [token_2_id[token] for token in doc]
	X = np.array(map(doc_2_id, processed_docs))
	return X, token_2_id, id_2_token

def labels_to_matrix(labels):
	"""
	converts a list of list of labels to numpy array of encoded uses. 
	uses sklearns label encoder

	Parameters
	----------
		
		labels(list): list of labels

	Returns
	---------
		encoded_labels (np.ndarray)
	"""
	
	encoder = preprocessing.LabelEncoder()
	y = encoder.fit_transform(np.array(labels))
	return y


def build_lstm(input_cardinality, input_length, embedding_size = 100, dropout = False, dropout_p = .5):
	"""
	returns a compiled keras lstm model.

	Parameters
	----------
		
		labels(list): list of labels

	Returns
	---------
		encoded_labels (np.ndarray)
	"""

	model = Sequential()
	model.add(Embedding(input_cardinality, embedding_size, input_length=input_length))
	model.add(LSTM(100))
	if dropout:
		model.add(Dropout(dropout_p))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

if __name__ == '__main__':

	max_words = 5000
	doc_max_len = 300
	model_path = 'trained_models/sentiment_model_og'
	model_info = 'trained_models/sentiment_model_og_info.pkl'

	nlp = spacy.load('en')
	docs, labels = collect_corpus('movie_reviews')
	docs = process_docs(docs)
	X, token_2_id, id_2_token = docs_to_seq_matrix(docs, max_words = max_words)
	y = labels_to_matrix(labels)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

	X_train = sequence.pad_sequences(X_train, doc_max_len)
	X_test = sequence.pad_sequences(X_test, doc_max_len)

	model = build_lstm(max_words, doc_max_len)
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=64)	

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	
	model_dict = {
		'token_2_id': token_2_id,
		'id_2_token': id_2_token
		}
	
	model.save(model_path)
	with open(model_info,'wb') as outfile:
		pickle.dump(model_dict, outfile)


