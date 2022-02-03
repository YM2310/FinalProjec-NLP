import json
from collections import Counter

import gensim
import nltk as nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.utils import tokenize
import gensim.downloader as api
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
def dataReader(file_name):
    with open(file_name, 'r') as corpus_file:
        corpus = [json.loads(jline) for jline in corpus_file.read().splitlines()]
    return corpus

def labels_to_ints(labels):
    convert_table = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    return [convert_table[label] for label in labels]


def model_res_to_labels(results):
    convert_table = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    labels=[]
    for res in results:
        index=np.argmax(res)
        labels.append(convert_table[index])
    return labels


def sentence2sequence(sentence,glove_model):
    sentence=sentence.lower()
    tokens = word_tokenize(sentence)
    sequence=[]
    for token in tokens:
        if token in glove_model.key_to_index:
            sequence.append(glove_model.key_to_index[token])
        else:
            sequence.append(0)
    return sequence



def prepare_data_for_model(data,glove):
    gold_labels = []
    sentences = []
    for line in data:
        if line["gold_label"] == '-':
            continue
        sentences.append(sentence2sequence(f"{line['sentence1']} {line['sentence2']}",glove))
        gold_labels.append(line['gold_label'])
    return sentences, gold_labels



def createPretrainedEmbeddingLayer(glove):
    vocab_len=len(glove)
    emb_dim=300
    embeddingMatrix = np.zeros((vocab_len, emb_dim))
    for word in glove.index_to_key:
        embeddingMatrix[glove.key_to_index[word],:]=glove[word]
    embeddingLayer = layers.Embedding(vocab_len, emb_dim, weights=[embeddingMatrix], trainable=False)
    return embeddingLayer

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

def testWithLSTM(train,glove):

    print("loaded glove")
    data_for_train = dataReader(train)
    cross_sections = []
    # word2vec=acquire_word_embedding()
    sentences, gold_labels = prepare_data_for_model(data_for_train,glove)
    print('data ready')
    num_chunks=8
    max_length = 50

    pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(glove)

    model = keras.models.Sequential()
    model.add(pretrainedEmbeddingLayer)
    model.add(layers.LSTM(128, dropout=0.3))
    model.add(layers.Dense(3, activation='softmax'))
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=0.001)  # weakspot- how to choose optimizer?
    metrics = ["accuracy"]
    model.summary()
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    sentences_chunks = list(chunks(sentences, num_chunks))
    labels_chunks = list(chunks(gold_labels, num_chunks))

    for i in range(num_chunks):
        train_sequences = []
        train_labels = []
        for j in range(num_chunks):
            if j != i:
                train_sequences.extend(sentences_chunks[j])
                train_labels.extend(labels_chunks[j])
        train_labels = np.array(labels_to_ints(train_labels))
        validation_sequences = sentences_chunks[i]
        validation_labels = np.array(labels_to_ints(labels_chunks[i]))

        train_padded = np.array(pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post'))
        validation_padded = np.array(
            pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post'))
        train_labels = keras.utils.to_categorical(train_labels, num_classes=3)
        validation_labels = keras.utils.to_categorical(validation_labels, num_classes=3)
        model.fit(train_padded, train_labels, epochs=10, validation_data=(validation_padded, validation_labels),
                  verbose=2)  # weakspot - what is verbose?
        print('training iteration: ',i)
    model.save('lstm_model5.h5')
    # TODO- Test on actual test
    # Implement a way to save the model once trained
    # improvement- see how to vectorize better- look at papers from SNLI.

def test(test_path,glove):
    max_length = 50

    model = load_model('lstm_model4.h5')
    print("loaded glove")
    data_for_test = dataReader(test_path)
    test_padded,golden_labels = prepare_data_for_model(data_for_test,glove)
    test_padded = np.array(pad_sequences(test_padded, maxlen=max_length, padding='post', truncating='post'))
    predictions = model.predict(test_padded)
    predictions_labels=model_res_to_labels(predictions)
    data= dataReader(test_path)
    gold_labels=[]
    for line in data:
        if line['gold_label']=='-':
            continue
        gold_labels.append(line['gold_label'])
    a = accuracy_score(gold_labels, predictions_labels)
    ConfusionMatrixDisplay.from_predictions(
        gold_labels, predictions_labels)
    plt.show()
    print(a)
    f1=f1_score(gold_labels,predictions_labels)
    return a,f1
if __name__ == '__main__':
    print("start")
    glove=api.load("glove-wiki-gigaword-300")
    test1=[1,2,3,4,5,6,7,8,9,10,11,12]
    test_c=list(chunks(test1,3))
    for i in range(len(test_c)):
        extended=[]
        for j in range(len(test_c)):
            if j!=i:
                extended.extend(test_c[j])
        small=test_c[i]
        print(extended,small)

    testWithLSTM('snli_1.0/snli_1.0_dev.jsonl',glove)
    predictions = test('snli_1.0/snli_1.0_test.jsonl',glove)

    # print(predictions)
    print("WOOHOO")
