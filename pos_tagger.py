from keras.models import load_model
import numpy as np
import pickle
from collections import defaultdict
import os

# for creating necessary mappings
def create_mappings():
    with open('mappings/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('mappings/idx_tag.pkl', 'rb') as f:
        idx_to_tag = pickle.load(f)
    # create word to index mapping
    # for every unknown word the dict will give index 1 which is <UNK>
    word_to_idx = defaultdict(lambda:1, {word:idx for idx,word in enumerate(vocab)})
    
    return word_to_idx, idx_to_tag


# converts the tokens to its numerical representation
# output: (m, max sequence length)
def convert_to_num(sentences, token_to_idx, pad=0, dtype='int32'):
    # find the max sentence length
    max_sent_len = max(map(len, sentences))
    
    # create the matrix
    mat = np.empty([len(sentences), max_sent_len], dtype)
    # fill with padding
    mat.fill(pad)
    
    # convert to numerical mappings
    for i, sentence in enumerate(sentences):
        num_row = [token_to_idx[token] for token in sentence]
        mat[i, :len(num_row)] = num_row
   
    return mat


# do prediction
def predict_tags(test_sent, word_to_idx, model, idx_to_tag):
    test_sent_token = np.array([[word for word in sent.split()] for sent in test_sent])
    test_sent_num = convert_to_num(test_sent_token, word_to_idx, pad=0)
    pred = model.predict(test_sent_num)

    for i in range(len(pred)):
        result = []
        
        for j in range(len(pred[i])):  
            index = pred[i,j,:].argmax()
            result.append((test_sent_token[0,j], idx_to_tag[index]))
        print('Prediction: ', result)
        print()

def main():
    # load model
    model = load_model('models/model.h5')
    # load necessary mappings
    word_to_idx, idx_to_tag = create_mappings()
    

    while(True):
        
        print('Part of Speech Tagger\n')
        test = []
        #test.append(input('Enter sentence: '))
        test.append('i like eating')
        predict_tags(test, word_to_idx, model, idx_to_tag)

        ch = input('Continue ? Y or N\n')
        if ch == 'n' or ch == 'N':
            return
        while(ch!= 'y' or ch!='Y'):
            ch = input('Invalid Choice Entered')

if __name__ == main():
    main()