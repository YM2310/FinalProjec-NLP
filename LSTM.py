import json
from collections import Counter

import gensim
import nltk as nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.utils import tokenize
import gensim.downloader as api
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




def acquire_word_embedding():
    return api.load("glove-wiki-gigaword-300")
def dataReader(file_name):
    with open(file_name,'r') as corpus_file:
        corpus = [json.loads(jline) for jline in corpus_file.read().splitlines()]
    return corpus

def counter_word(text_list):
    count = Counter()
    for text in text_list:
        for word in text.split():
            count[word] += 1
    return count

def labels_to_ints(labels):
    convert_table={
        "entailment":0,
        "neutral":1,
        "contradiction":2
    }

    return [convert_table[label] for label in labels]

def prepair_data_for_model(data):
    gold_labels=[]
    sentences=[]
    for line in data:
        if line["gold_label"] == '-':
            continue
        sentences.append(f"{line['sentence1']} {line['sentence2']}")
        gold_labels.append(line['gold_label'])
    return sentences, gold_labels

def testWithLSTM(train, test):
    data_for_train=dataReader(train)
    cross_sections=[]
    #word2vec=acquire_word_embedding()
    sentences,gold_labels=prepair_data_for_model(data_for_train)
    counter=counter_word(sentences)
    num_unique_words=len(counter)
    trainsplit=int(len(data_for_train)*0.7)
    train_sentences=sentences[:trainsplit]
    train_labels=np.array(labels_to_ints(gold_labels[:trainsplit]))
    validation_sentences=sentences[trainsplit:]
    validation_labels=np.array(labels_to_ints(gold_labels[trainsplit:]))
    tokenizer=Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences=tokenizer.texts_to_sequences(train_sentences)
    validation_sequences=tokenizer.texts_to_sequences(validation_sentences)

    max_length=50
    train_padded=np.array(pad_sequences(train_sequences,maxlen=max_length,padding='post',truncating='post'))
    validation_padded=np.array(pad_sequences(validation_sequences,maxlen=max_length,padding='post',truncating='post'))


    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words,32,input_length=max_length))
    model.add(layers.LSTM(128,dropout=0.3))
    model.add(layers.Dense(3,activation='softmax'))
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(lr=0.001) # weakspot- how to choose optimizer?
    metrics = ["accuracy"]
    model.summary()
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    train_labels=keras.utils.to_categorical(train_labels,num_classes=3)
    validation_labels=keras.utils.to_categorical(validation_labels,num_classes=3)
    model.fit(train_padded,train_labels,epochs=10,validation_data=(validation_padded,validation_labels),verbose=2)# weakspot - what is verbose?

    predictions=model.predict(validation_padded)
    #TODO- Test on actual test
    # Implement a way to save the model once trained
    # improvement- see how to vectorize better- look at papers from SNLI.
    print("WOOHOO")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testWithLSTM('snli_1.0/snli_1.0_dev.jsonl','snli_1.0/snli_1.0_test.jsonl')