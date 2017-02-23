from keras.preprocessing import sequence
import numpy as np


#create the processing function
def process(document, nlp = None):
    """
    Parameters
    ----------
    document: str
        The document we want to process
        
    Returns
    ----------
    processed_doc: string
    """
    if not nlp:
        return document.lower()

    #create spacy object
    spacy_doc = nlp(unicode(document), parse=False, entity=False)
    
    #grab the lemma for each token in the document
    processed_tokens = map(lambda token: token.lower_, spacy_doc)
    
    #join lemmas to a string
    result = processed_tokens

    return result



def query_to_doc_array(doc, token_2_id, nlp, input_length):
    
    tokens = process(doc, nlp)
    vector = []
    for token in tokens:
        if token in token_2_id:
            vector.append(token_2_id[token] )
    vector = [vector]
    return sequence.pad_sequences(vector, input_length)
    