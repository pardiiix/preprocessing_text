import json
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
from contractions import contractions_dict
from autocorrect import Speller
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import asarray, array, zeros
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import keras.metrics
from keras import backend as K
# import autocorrect
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def to_lower(text):
    """
    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".
    """
    # return ' '.join([w.lower() for w in nltk.word_tokenize(text)])
    lower_text = df['comments'].str.lower()

def strip_punctuation(text):
    """
    Removinmg puctuation
    """
    return ''.join(c for c in text if c not in punctuation)

def deEmojify(inputString):
    '''
    Removing emojis from text
    '''
    return inputString.encode('ascii', 'ignore').decode('ascii')


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def prep_text():

    #reading the labelled comment file
    with open('SD', 'r') as file:
        sd = file.readlines()

    with open('no_SD', 'r') as file:
        no_sd = file.readlines()
        # lower_text = []
        # data = json.load(json_file)
        # for i in data["memory_loss"]:
        #     lower_text.append(to_lower(i["comments"])) #converting the comments in memory_loss dictionary to lower case

    df = pd.DataFrame(columns=['comments', 'polarity'])
    df['comments'] = no_sd + sd
    df['polarity'] = [0] * len(no_sd) + [1] * len(sd)
    df = df.sample(frac=1, random_state = 10) #shuffling the rows
    df.reset_index(inplace = True, drop = True)
    df['comments'] = df['comments'].str.lower() #Converting text to lower case


    stopword = nltk.corpus.stopwords.words("english")
    speller = Speller()


    for i in range(len(df['comments'])):
        df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
        df['comments'][i] = deEmojify(df['comments'][i])
        df['comments'][i] = strip_punctuation(df['comments'][i])
        df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in stopword) #removing stopwords and spell-correcting



    max_sent_len = 40
    max_vocab_size = 200
    word_seq = [text_to_word_sequence(comment) for comment in df['comments']]
    # print(word_seq)

    # vectorizing a text corpus, turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    tokenizer = Tokenizer(num_words = max_vocab_size)
    tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq]) #Updates internal vocabulary based on a list of texts up to the max_sent_len.
    # print("vocab size: ", len(tokenizer.word_index)) #vocab size: 949

    #converting sequence of words to sequence of indices
    X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
    X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')

    y = df['polarity']
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10, test_size=0.1)
    X_train, X_dev, y_train, y_dev = train_test_split(X_test,y_test, random_state=10, test_size=0.5)

    #creating a dictionary for glove such that embeddings_dictionary[word] = word_vector
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt')
    for line in glove_file:
        records = line.split()
        word = records[0]
        word_vector = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = word_vector
    glove_file.close()


    # print(tokenizer.word_index)

    #creating an embedding matrix with words in our vocabulary and word vectors in glove
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    #building a sequential model by stacking neural net units
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size,
                        output_dim = 100,
                        weights = [embedding_matrix],
                        input_length = max_sent_len,
                        trainable = False,
                        name = 'word_embedding_layer',
                        mask_zero=True
                        ))


    model.add(LSTM(64, return_sequences=False, name = 'lstm_layer'))
    model.add(Dense(1, activation= 'sigmoid', name = 'output_layer'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', recall_m, precision_m, f1_m])
    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose = 1, validation_split =0.2) #verbose =1 : see trainig progress for each epoch

    score = model.evaluate(X_dev, y_dev, verbose = 1)

    # print(score)
    print("Dev set score: ", score[0])
    print("Dev set accuracy: ", score[1])

    # print(history.history)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()

    # #lemmatizer
    # wordnet_lemmatizer = WordNetLemmatizer()
    # lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in correct_spelling]
    # # print(lemmatized_word)

    # #stemming
    # snowball_stemmer = SnowballStemmer("english")
    # stemmed_word = [snowball_stemmer.stem(word) for word in lemmatized_word]
    # return stemmed_word

    # x = re.findall("[^a-zA-Z]very$", ' '.join(c for c in df))
    # print(x)
    # print((df))
prep_text()