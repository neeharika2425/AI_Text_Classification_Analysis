import sys
import random
import time
import math
import numpy as np
import codecs

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
bi_vectorizer = CountVectorizer(ngram_range=(2, 2))
tfidf_transformer_uni = TfidfTransformer()
tfidf_transformer_bi = TfidfTransformer()

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
def getFeatureVectors(files,flag,conf):
	uni_vectors = []
	bi_vectors = []
	corpus = []
        for docname in files:
			string = ''
                        textDoc = codecs.open(docname, "r",encoding='utf-8', errors='ignore').readlines()
                        for line in textDoc:
                                line = line.rstrip("\n").rstrip("\r").rstrip("\t")
				if conf == 1:
					line = removeStopwords(line)
					line = stemmer(line)
                                words = line.split()
                                if len(words) != 0:
                                        if (words[0] != "From:" and words[0] != "Subject:" and words[0] != "Organization:" and words[0] != "Lines:"):
						string = string + " " + line

         		corpus.append(string)
	if flag == "train":
		
		if conf == 1:
			uni_vectors = uni_vectorizer.fit_transform(corpus)
			best_vectors = tfidf_transformer_uni.fit_transform(uni_vectors).toarray()
			best_vectors.tolist()
			print len(best_vectors[0])
			return best_vectors
		else:
			uni_vectors = uni_vectorizer.fit_transform(corpus).toarray()	
			uni_vectors = tfidf_transformer_uni.fit_transform(uni_vectors).toarray()
			bi_vectors = bi_vectorizer.fit_transform(corpus).toarray()	
			bi_vectors = tfidf_transformer_bi.fit_transform(bi_vectors).toarray()
        		bi_vectors.tolist()
			uni_vectors.tolist()
			print len(uni_vectors[0])
			return uni_vectors,bi_vectors
	if flag == "test":
		
		if conf == 1:
			uni_vectors = uni_vectorizer.transform(corpus)	
			best_vectors = tfidf_transformer_uni.transform(uni_vectors).toarray()
			best_vectors.tolist()
			return best_vectors
		else:
			uni_vectors = uni_vectorizer.transform(corpus).toarray()	
			uni_vectors = tfidf_transformer_uni.transform(uni_vectors).toarray()
			bi_vectors = bi_vectorizer.transform(corpus).toarray()	
			bi_vectors = tfidf_transformer_bi.transform(bi_vectors).toarray()
        		bi_vectors.tolist()
			uni_vectors.tolist()
			return uni_vectors,bi_vectors


def bestConfiguration(train_vec,test_vec,labels):
#	train_new = SelectKBest(chi2, k=1000).fit_transform(train_vec, labels)
#	test_new = SelectKBest(chi2, k=1000).transform(test_vec)
	clf = SGDClassifier(penalty='l2',alpha = 0.2e-3,loss='hinge')
	return clf.fit(train_vec, labels).predict(test_vec)
#	clf = SVC(kernel='poly')
#	return clf.fit(train_vec, labels).predict(test_vec)
def logisticRegression(train,test,labels):
	lr = LogisticRegression()	
	return lr.fit(train, labels).predict(test) 

def randomForest(train,test,labels):
	rf = RandomForestClassifier()	
	return rf.fit(train, labels).predict(test) 

def SVM(train,test,labels):
	#clf = SVC()	
	clf = SGDClassifier(penalty='none',loss='hinge')
	return clf.fit(train, labels).predict(test) 

def naiveBayes(train,test,labels):
	gnb = MultinomialNB()
	return gnb.fit(train, labels).predict(test)

if __name__ == '__main__':
	folder_train = sys.argv[1]
	folder_test = sys.argv[2]
	flag = int(sys.argv[3])
	#folder_train = "Selected 20NewsGroup/Training/"
	#folder_test = "Selected 20NewsGroup/Test/"
	
	dataFiles_train = load_files(folder_train,shuffle=True)
	dataFiles_test  = load_files(folder_test,shuffle=True)
	train_labels = dataFiles_train.target
	test_labels = dataFiles_test.target

	if flag == 1:
		bf_train = getFeatureVectors(dataFiles_train.filenames,"train",1)
		bf_test = getFeatureVectors(dataFiles_test.filenames,"test",1)
	
		#print bf_train
		bf_pred = bestConfiguration(bf_train,bf_test,train_labels)

		acc = precision_recall_fscore_support(test_labels, bf_pred, average='macro')
		print 'For best configuration\n' +  'precision %f , recall  %f , F1 score %f' %(acc[0],acc[1],acc[2])   
	if flag == 0:
		uni_train,bi_train = getFeatureVectors(dataFiles_train.filenames,"train",0)
		uni_test,bi_test = getFeatureVectors(dataFiles_test.filenames,"test",0)
		size = [0.2,0.4,0.5,0.6,0.7,0.8,1]
		precision_uni = []
		recall_uni = []
		f1_uni = []
		precision_bi = []
		recall_bi = []
		f1_bi = []
		f1_nb = []
		f1_svm = []
		f1_lr = []
		f1_rf = []
		train_data = []
		train_data = [x * len(uni_train) for x in size]
		for s in size:
			uni_train_part,bi_train_part,train_labels_part = generateData(uni_train,bi_train,train_labels,s)
			#uni_train_part,bi_train_part,train_labels_part = uni_train[:int(s)],bi_train[:int(s)],train_labels[:int(s)]
			uni_gnb_pred = naiveBayes(uni_train_part,uni_test,train_labels_part)	
			bi_gnb_pred = naiveBayes(bi_train_part,bi_test,train_labels_part)	
			acc = precision_recall_fscore_support(test_labels, uni_gnb_pred, average='macro')
			f1_nb.append(acc[2])
			if s == 1:
				precision_uni.append(acc[0])
				recall_uni.append(acc[1])
				f1_uni.append(acc[2])
				acc = precision_recall_fscore_support(test_labels, bi_gnb_pred, average='macro')
				precision_bi.append(acc[0])
				recall_bi.append(acc[1])
				f1_bi.append(acc[2])
		
			uni_gnb_pred = logisticRegression(uni_train_part,uni_test,train_labels_part)	
			bi_gnb_pred = logisticRegression(bi_train_part,bi_test,train_labels_part)
			acc = precision_recall_fscore_support(test_labels, uni_gnb_pred, average='macro')
			f1_lr.append(acc[2])
			if s == 1:
				precision_uni.append(acc[0])
				recall_uni.append(acc[1])
				f1_uni.append(acc[2])
				acc = precision_recall_fscore_support(test_labels, bi_gnb_pred, average='macro')
				precision_bi.append(acc[0])
				recall_bi.append(acc[1])
				f1_bi.append(acc[2])

			uni_gnb_pred = SVM(uni_train_part,uni_test,train_labels_part)	
			bi_gnb_pred = SVM(bi_train_part,bi_test,train_labels_part)
			acc = precision_recall_fscore_support(test_labels, uni_gnb_pred, average='macro')
			f1_svm.append(acc[2])
			if s == 1:
				precision_uni.append(acc[0])
				recall_uni.append(acc[1])
				f1_uni.append(acc[2])
				acc = precision_recall_fscore_support(test_labels, bi_gnb_pred, average='macro')
				precision_bi.append(acc[0])
				recall_bi.append(acc[1])
				f1_bi.append(acc[2])
		
			uni_gnb_pred = randomForest(uni_train_part,uni_test,train_labels_part)	
			bi_gnb_pred = randomForest(bi_train_part,bi_test,train_labels_part)
			acc = precision_recall_fscore_support(test_labels, uni_gnb_pred, average='macro')
			f1_rf.append(acc[2])
			if s == 1:
				precision_uni.append(acc[0])
				recall_uni.append(acc[1])
				f1_uni.append(acc[2])
				acc = precision_recall_fscore_support(test_labels, bi_gnb_pred, average='macro')
				precision_bi.append(acc[0])
				recall_bi.append(acc[1])
				f1_bi.append(acc[2])

		algos = ['Naive Bayes','Logistic Regression','SVM','RandomForests']
		print '*****Unigrams*****'
		for i in range(len(algos)):
			print 'For ' + algos[i] + '\nprecision %f , recall  %f , F1 score %f' %(precision_uni[i],recall_uni[i],f1_uni[i])   
		print '/n*****Bigrams*****'
		for i in range(len(algos)):
			print 'For ' + algos[i] + '\nprecision %f , recall  %f , F1 score %f' %(precision_bi[i],recall_bi[i],f1_bi[i])   
		plt.plot(train_data,f1_nb,label = 'Naive Bayes')
		plt.plot(train_data,f1_lr,label = 'Logistic Regression')
		plt.plot(train_data,f1_svm,label = 'Svm')
		plt.plot(train_data,f1_rf,label = 'Random Forest')
		plt.ylabel('F1 Score')
        	plt.xlabel('Training Data Size')
   		plt.legend(loc='lower right')
		plt.show()
