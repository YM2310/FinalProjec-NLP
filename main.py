import json

import gensim
import nltk as nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import word_tokenize
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


def acquire_word_embedding():
    return api.load("glove-wiki-gigaword-300")
def dataReader(file_name):
    with open(file_name,'r') as corpus_file:
        corpus = [json.loads(jline) for jline in corpus_file.read().splitlines()]
    return corpus

def n_gram_cross_section(sentence1_words, sentence2_words, n_gram):
    cross_section=[]
    for word in sentence1_words:
        if word in sentence2_words and word not in cross_section:
            cross_section.append(word)
    return cross_section
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def n_gram_cross_section_with_word2vec(word2vec,sentence1_words, sentence2_words):
    cross_section=set()
    for word in sentence1_words:
        if word in sentence2_words:
            cross_section.add(word)
        else:
            for word2 in sentence2_words:
                try:
                    if cosine(word2vec[word],word2vec[word2])>0.9:
                        cross_section.add(word2)
                except:
                    a=1
    return cross_section

def tokenize_and_remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))

    word_tokens = tokenize(sentence)

    return [w.lower() for w in word_tokens if not w.lower() in stop_words]

def get_verbs_from_tree(tree_as_string):
    tree = nltk.tree.Tree.fromstring(tree_as_string)
    verbs=[]
    for s in tree.subtrees(lambda t: t.height() == 2):
        if s.label().startswith('V'):
            verbs.append(s[0])
    return verbs
def prepareDataForClassifier(line,word2vec):
    #sentence1=line['sentence1_parse'].lower()
    #sentence2=line['sentence2_parse'].lower()
    sentence1_words=tokenize_and_remove_stopwords(line['sentence1'])
    sentence2_words=tokenize_and_remove_stopwords(line['sentence2'])
    #sentence1_words=nltk.tree.Tree.fromstring(sentence1).leaves()
    #sentence2_words=nltk.tree.Tree.fromstring(sentence2).leaves()
    return (" ".join(n_gram_cross_section_with_word2vec(word2vec,sentence1_words,sentence2_words)))


def testWithClassifier(train, test):
    data_for_train=dataReader(train)
    cross_sections=[]
    word2vec=acquire_word_embedding()
    gold_labels=[]
    count_vectorized = CountVectorizer(max_features=1000,
                                       strip_accents='ascii',
                                       lowercase=True)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    prec=-1
    for i,line in enumerate(data_for_train):
        if line['gold_label'] == "-":
            continue
        if i%(len(data_for_train)/100.0)==0:
            prec+=1
            print(f"{prec}%")
        cross_sections.append(prepareDataForClassifier(line,word2vec))
        gold_labels.append(line['gold_label'])
    print("acuierd data")
    x_train_counts = count_vectorized.fit_transform(cross_sections)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    classifier=LogisticRegression(max_iter=1000).fit(x_train_tfidf,gold_labels)
    print("trained")

    data_for_test=dataReader(test)
    cross_sections_test=[]
    gold_labels_test=[]
    for line in data_for_test:
        if line['gold_label'] == "-":
            continue
        cross_sections_test.append(prepareDataForClassifier(line,word2vec))
        gold_labels_test.append(line['gold_label'])
    x_new_counts = count_vectorized.transform(cross_sections_test)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)
    print("predicting")
    predicted = classifier.predict(x_new_tfidf)
    a=accuracy_score(gold_labels_test,predicted)
    ConfusionMatrixDisplay.from_predictions(
    gold_labels_test,predicted)
    plt.show()
    f1 = f1_score(gold_labels_test, predicted, average=None)
    print(f1)
    print(a)
    print("done")



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testWithClassifier('snli_1.0/snli_1.0_train.jsonl','snli_1.0/snli_1.0_test.jsonl')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
