# -*- coding: utf-8 -*-
from sklearn.datasets import load_files
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics
from sklearn import svm

movie_reviews = load_files('txt_sentoken_train')
X_train= movie_reviews.data 
y_train=movie_reviews.target 
movie_reviews2=load_files('txt_sentoken_test')
X_test=movie_reviews2.data
y_test=movie_reviews2.target
count_vect = CountVectorizer(ngram_range=(2,2),max_features=5000)
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts=count_vect.transform(X_test)

lin_svc= svm.LinearSVC(C=2).fit(X_train_counts, y_train)
#svc=svm.SVC(C=0.7,kernel='linear').fit(X_train_counts,y_train)
rbf_svc=svm.SVC(gamma=0.7,C=2,kernel='rbf').fit(X_train_counts,y_train)
lsvc_predict = lin_svc.predict(X_test_counts)
#svc_predict=svc.predict(X_test_counts)
rbf_predict=rbf_svc.predict(X_test_counts)  
print 'Linear SVC:'
print 'Accuracy: ',np.mean(lsvc_predict==y_test)*100
print (metrics.classification_report(y_test, lsvc_predict))
#print 'Linear Kernel SVC:'
#print 'Accuracy: ',np.mean(svc_predict==y_test)*100
#print(metrics.classification_report(y_test, svc_predict))
print 'RBF Kernel SVC:'
print 'Accuracy: ',np.mean(rbf_predict==y_test)*100 
print(metrics.classification_report(y_test, rbf_predict))