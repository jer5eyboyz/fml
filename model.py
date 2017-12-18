import re
import nltk
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
from scipy import stats
import _pickle as cPickle
from nltk.corpus import stopwords
from scipy.stats import itemfreq
import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)
warnings.filterwarnings("ignore", category= FutureWarning)

 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix


def splitData(data):
 
    # Y = data['class'].values
    # data = data.drop(['class'], axis = 1)
    # X = data.values
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    print ("Data size:", data.shape)
    print (list(data.columns.values))

    Y_downsampled = data['class'].values
    traindf_downsampled = data.drop(['class'], axis = 1)
    X_downsampled = traindf_downsampled.values
    X_train, X_test, Y_train, Y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.3)
    
    return X_train, X_test, Y_train, Y_test


def classiferCompare(X_train, X_test, Y_train, Y_test):
    names = [ 
            "KNeighborsClassifier", 
            "Linear SVM",
            # "RBF SVM",
            "Decision Tree",
            "Stochastic Gradient Descent",
            "Gaussian Process", 
            "LDA", 
            "QDA",
            "Random Forest",
            "GaussianNB",
            "AdaBoost", 
            "XGBoost", 
            "LogisticRegression(L1)", "LogisticRegression(L2)"
            ]

    classifiers = [
        KNeighborsClassifier(3),
        LinearSVC(C=1,penalty='l1', loss='squared_hinge', dual=False),
        # SVC(kernel='rbf', C=1000),
        DecisionTreeClassifier(),
        SGDClassifier(loss="perceptron", penalty="l2"),
        GaussianProcessClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(n_estimators=200, max_features=15),
        GaussianNB(),
        AdaBoostClassifier(),
        LogisticRegression(penalty='l1'),
        LogisticRegression(penalty='l2')
        ]

    figure = plt.figure(figsize=(27, 9))

    # iterate over classifer models
    print ("Start training!")

    plot_number = 1
    for name, clf in zip(names, classifiers):
        #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        y_score = clf.fit(X_train, Y_train)
        train_score = cross_val_score(clf, X_train, Y_train, cv=5)
        test_score = clf.score(X_test, Y_test)
        Y_pred = clf.predict(X_test)

        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        
        print ("***", name, "***")
        print ("Train Score:", train_score.mean())
        print ("Test Score:", test_score)
        print ("Precision:", precision)
        print ("Recall:", recall)
        print ("F1 score", f1)
        print (classification_report(Y_test,Y_pred))
        
        # Plot ROC curve
        ax = plt.subplot(4, len(classifiers)/2, plot_number)
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(name)
        plt.xlabel('False positive rate', fontsize=12)
        plt.ylabel('True positive rate', fontsize=12)
        plt.legend(loc="lower right")
        plot_number += 1

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    

def paramSearchLogistic(X_train, X_test, Y_train, Y_test):

    print ("Logistic Parameter Search")
    
    parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty':['l1'],
                  #'solver':['liblinear'],
                  'dual':[False]
                  }
    LR1 = LogisticRegression()
    clf = GridSearchCV(LR1, param_grid=parameters)
    clf.fit(X_train, Y_train)
    print (clf.best_estimator_)
    print (clf.best_params_)
    print (clf.best_score_)

    parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty':['l2'],
                  'solver':['newton-cg','lbfgs','sag'],
                  'dual':[False]
                  }
    LR2 = LogisticRegression()
    clf = GridSearchCV(LR2, param_grid=parameters)
    clf.fit(X_train, Y_train)
    print (clf.best_estimator_)
    print (clf.best_params_)
    print (clf.best_score_)


def paramSearchSVM(X_train, X_test, Y_train, Y_test):

    print ("LinearSVC Parameter Search")
    
    parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty':['l2'],
                  'loss':['squared_hinge'],
                  'dual':[False]
                  }
    svc = LinearSVC()
    clf = GridSearchCV(svc, param_grid=parameters)
    clf.fit(X_train, Y_train)
    print (clf.best_estimator_)
    print (clf.best_params_)
    print (clf.best_score_)



def paramSearchAdaboost(X_train, X_test, Y_train, Y_test):
    
    print ("Adaboost Parameter Search")
    # # Decision Tree
    # print ("Base estimator: Decision Tree")
    # parameters = {'base_estimator__splitter':["best"],
    #               'base_estimator__min_samples_leaf': [1, 5, 10], 
    #               'base_estimator__max_depth': [1, 5, 10, 30],
    #               # 'base_estimator__min_leaf_size':[10, 15, 40],
                    # min size fo split
                    # minimal gain
    #               'n_estimators': [50, 100, 200, 250, 300], 
    #               }
    # adaBoost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
    # clf = GridSearchCV(adaBoost, param_grid=parameters)
    # clf.fit(X_train, Y_train)
    # print (clf.best_estimator_)
    # print (clf.best_params_)
    # print (clf.best_score_)


    # Random Forest
    print ("Base estimator: Random Forest ")
    parameters = {'base_estimator__n_estimators':[10, 25, 50, 100, 150, 200],
                  'base_estimator__min_samples_split':[2, 5, 10],
                  'base_estimator__max_features':[5, 10, 20, 30, 40],
                  'n_estimators': [50, 100, 200, 250, 300], 
                  }
    adaBoost = AdaBoostClassifier(base_estimator = RandomForestClassifier())
    clf = GridSearchCV(adaBoost, param_grid=parameters)
    clf.fit(X_train, Y_train)
    print (clf.best_estimator_)
    print (clf.best_params_)
    print (clf.best_score_)
    

    # # Linear model with Gaussian Naive Bayes  
    # print ("Base estimator: Gaussian Naive Bayes")
    # parameters = {
    #               'n_estimators': [100, 200, 250, 300, 350, 400], 
    #               }
    # adaBoost = AdaBoostClassifier(base_estimator = GaussianNB())
    # clf = GridSearchCV(adaBoost, param_grid=parameters)
    # clf.fit(X_train, Y_train)
    # print (clf.best_estimator_)
    # print (clf.best_params_)
    # print (clf.best_score_)


    # Linear model with LinearSVM L1
    # print ("Base estimator: LinearSVM (L1)")
    # parameters = {'base_estimator__penalty':['l1'],
    #               'base_estimator__loss':['squared_hinge'],
    #               'base_estimator__dual':[False],
    #               'base_estimator__C':[0.01, 0.1, 1, 10, 100],
    #               'n_estimators': [50, 100, 200, 250, 300], 
    #               }
    # adaBoost = AdaBoostClassifier(base_estimator = LinearSVC(), algorithm='SAMME')
    # clf = GridSearchCV(adaBoost, param_grid=parameters)
    # clf.fit(X_train, Y_train)
    # print (clf.best_estimator_)
    # print (clf.best_params_)
    # print (clf.best_score_)


def main():

    data = pd.read_pickle('1995_2017_down_sample_30_feature.p')
    # data = pd.read_pickle('2010_2017_down_sample.p')
    X_train, X_test, Y_train, Y_test = splitData(data)
    classiferCompare(X_train, X_test, Y_train, Y_test)
    # data = pd.read_pickle("~/Desktop/fml/2010_2017_down_sample.p")

    # Parameter Search for Adaboost
    # paramSearchAdaboost(X_train, X_test, Y_train, Y_test)

    # Parameter Search for Logistic Regression(L1)
    # paramSearchLogistic(X_train, X_test, Y_train, Y_test)

    # Parameter Search for Logistic Regression(L1)
    # paramSearchSVM(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":

    main()

