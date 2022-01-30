import json

import nltk as nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.utils import tokenize




# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
def dataReader(file_name):
    with open(file_name,'r') as corpus_file:
        corpus = [json.loads(jline) for jline in corpus_file.read().splitlines()]
    return corpus

def n_gram_cross_section(sentence1_words, sentence2_words, n_gram):
    cross_section=[]
    for word in sentence1_words:
        if word in sentence2_words:
            cross_section.append(word)
    return cross_section


def get_verbs_from_tree(tree_as_string):
    tree = nltk.tree.Tree.fromstring(tree_as_string)
    verbs=[]
    for s in tree.subtrees(lambda t: t.height() == 2):
        if s.label().startswith('V'):
            verbs.append(s[0])
    return verbs
def prepareDataForClassifier(line):
    sentence1=line['sentence1_parse']
    sentence2=line['sentence2_parse']
    sentence1_words=nltk.tree.Tree.fromstring(sentence1).leaves()
    sentence2_words=nltk.tree.Tree.fromstring(sentence2).leaves()
    cross=n_gram_cross_section(sentence1_words,sentence2_words,1)




def testWithClassifier(train, test):
    data_for_train=dataReader(train)
    for line in data_for_train:
        prepareDataForClassifier(line)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testWithClassifier('snli_1.0/snli_1.0_train.jsonl','snli_1.0/snli_1.0_test.jsonl')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
