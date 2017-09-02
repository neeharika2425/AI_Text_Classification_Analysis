import sys
import random
import time
import math
import numpy as np
import codecs
import pickle

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from collections import Iterable
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from nltk.corpus import stopwords
from nltk.stem.porter import *

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def generateData(uni_vecs,bi_vecs,labels,s):
        uni_vecs_part = []
        bi_vecs_part = []
        labels_part = []
        randints = []
        k = int(s * len(uni_vecs))
        #randomTestData(len(totalVec),0,k,randints)
        randints = random.sample(xrange(len(uni_vecs)),k)
        for i in range(len(uni_vecs)):
                if i in randints:
                        uni_vecs_part.append(uni_vecs[i])
                        bi_vecs_part.append(bi_vecs[i])
                        labels_part.append(labels[i])
        return uni_vecs_part,bi_vecs_part,labels_part

def removeStopwords(string):
    	cachedStopWords = stopwords.words("english")
        return ' '.join([word for word in string.split() if word not in cachedStopWords])
def stemmer(string):
	return ' '.join(PorterStemmer().stem_word(word) for word in string.split(" "))
def getFeatureVectors(files,models):
	uni_vectors = []
	bi_vectors = []
	corpus = []
        for docname in files:
			string = ''
                        textDoc = codecs.open(docname, "r",encoding='utf-8', errors='ignore').readlines()
                        for line in textDoc:
                                line = line.rstrip("\n").rstrip("\r").rstrip("\t")
				line = removeStopwords(line)
				line = stemmer(line)
                                words = line.split()
                                if len(words) != 0:
                                        if (words[0] != "From:" and words[0] != "Subject:" and words[0] != "Organization:" and words[0] != "Lines:"):
						string = string + " " + line

        		corpus.append(string)
	uni_vectors = models[0].transform(corpus)	
	best_vectors = models[1].transform(uni_vectors).toarray()
	best_vectors.tolist()
	return best_vectors


if __name__ == '__main__':
	folder_test = sys.argv[1]
	#folder_train = "Selected 20NewsGroup/Training/"
	#folder_test = "Selected 20NewsGroup/Test/"
	with open('model_.pickle','rb') as f:
                models = pickle.load(f)

	dataFiles_test  = load_files(folder_test,shuffle=True)
	test_labels = dataFiles_test.target
	
	bf_test = getFeatureVectors(dataFiles_test.filenames,models)
	bf_pred = models[2].predict(bf_test)

	acc = precision_recall_fscore_support(test_labels, bf_pred, average='macro')
	print 'For best configuration\n' +  'precision %f , recall  %f , F1 score %f' %(acc[0],acc[1],acc[2])   
