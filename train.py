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

uni_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def removeStopwords(string):
    	cachedStopWords = stopwords.words("english")
        return ' '.join([word for word in string.split() if word not in cachedStopWords])
def stemmer(string):
	return ' '.join(PorterStemmer().stem_word(word) for word in string.split(" "))
def getFeatureVectors(files):
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
	uni_vectors = uni_vectorizer.fit_transform(corpus)
	best_vectors = tfidf_transformer.fit_transform(uni_vectors).toarray()
	best_vectors.tolist()
	return best_vectors

def bestConfiguration(train_vec,labels):
        clf = SGDClassifier(penalty='l2',alpha = 0.2e-3,loss='hinge')
        #clf = SGDClassifier(penalty='none',loss='hinge')
        return clf.fit(train_vec,labels)

if __name__ == '__main__':
	folder_train = sys.argv[1]
	#folder_train = "Selected 20NewsGroup/Training/"
	#folder_test = "Selected 20NewsGroup/Test/"
	models = []
	models.append(uni_vectorizer)
	models.append(tfidf_transformer)
	
	dataFiles_train = load_files(folder_train,shuffle=True)
	train_labels = dataFiles_train.target

	bf_train = getFeatureVectors(dataFiles_train.filenames)
	bf_model = bestConfiguration(bf_train,train_labels)
	models.append(bf_model)	
	with open('model_.pickle','wb') as f:
                              pickle.dump(models,f)
	print "MODEL TRAINED - Please check model_.pickle "

