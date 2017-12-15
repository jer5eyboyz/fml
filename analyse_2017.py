import re
import nltk
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
from scipy import stats
from sklearn import svm
from nltk.corpus import stopwords
from scipy.stats import itemfreq
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix




def bagwords (df):     
    
    #listname = ['artist_genres', 'artist_name', 'album_name', 'song_name']
    listname = ['artist_genres']
    dicname = {name: [] for name in listname}
    dicname['song_id'] = []
    listid = []
    dicdf = {}
    vol = {}

    for name in listname:
        
        for idx in range(df[name].size):
        
            l = df[name][idx]
            r = names_to_words(l)
            dicname[name].append(r) 
         
        vectorizer = CountVectorizer(analyzer='word',max_features=30)
        feature = vectorizer.fit_transform(dicname[name]).toarray().tolist()
        vol[name] = vectorizer.get_feature_names()    
        dicdf[name] = pd.DataFrame({name: feature}) 
        dicdf[name] = dicdf[name][name].apply(pd.Series)
        l = len(dicdf[name].columns)        
        dicdf[name].columns = [(name + '_' +vol[name][x]) for x in range(l)]
     
    for idx in range(df['song_id'].size):
        sid = df['song_id'][idx]
        listid.append(sid)
        dicdf['song_id'] = pd.DataFrame({'song_id':listid}) 
 
    result = pd.concat(dicdf.values(), axis =1)
    
    return result



def names_to_words(names):
    words = re.sub("[^a-zA-Z0-9]"," ",names).lower().split()
    words = [i for i in words if i not in set(stopwords.words("english"))]
    ## Need join as string for countvectorizer!
    return (" ".join(words))


def data_processing(fileName):

    ## general dataframe transform
    print('Start!')
    print('Reading data:', fileName) 
    df = pd.read_csv(fileName, sep=",", encoding = "ISO-8859-1")
    df = df.dropna()
    df = df.drop(['album_genres'],axis =1 )
    df['explicit'] = df['explicit'].apply(lambda x:1 if x == True else 0 ).astype(int)
    
    # Classify popularity by top 20% /80%
    popular = df['popularity'].quantile(0.8)
    df['class'] = df['popularity'].apply(lambda x: 1 if x >= popular else 0)
    df = df[(df.astype(str)['artist_genres'] != '[]')].reset_index()
    # df['year'] = [x.split('-')[0] for x in df['album_release_date']]

    df1 = bagwords(df)
    df = df.merge(df1, on='song_id', how='outer')
    df = df.drop(['Unnamed: 0', 'song_id', 'artist_id','album_id','song_name',
            'artist_name','album_name','uri', 'type', 'track_href',
            'analysis_url','artist_genres','album_release_date','popularity',
            'index'], axis=1)    

    print ("Number of Features:", len(df.columns.values))
    print ("Features:", list(df.columns.values))
    print( "Data size:", df.shape)

    return df


def train_model_test(data):
    Y = data['class'].values
    data = data.drop(['class'], axis = 1)
    X = data.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,Y_train)
    Ascores_Train = cross_val_score(clf, X_train, Y_train, cv=5)
    Ascores_Test = clf.score(X_test,Y_test)

    # print ("Random Forest Classifier")
    # print(Ascores_Train.mean()) 
    # print(Ascores_Train.std()) 
    # print(Ascores_Test)

    return Ascores_Test


def classiferCompare(data):

    figure = plt.figure(figsize=(27, 9))
    Y = data['class'].values
    data = data.drop(['class'], axis = 1)
    X = data.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    names = [ 
            "KNeighborsClassifier", 
            # "Linear SVM", "Poly SVM", "RBF SVM",
            # "Decision Tree",
            "Gaussian Process", "LDA", "QDA",
            "Random Forest", 
            "AdaBoost", "XGBoost", 
            "LogisticRegression(L1)", "LogisticRegression(L2)"
            ]

    classifiers = [
        KNeighborsClassifier(3),
        # SVC(kernel="linear"),
        # SVC(kernel="poly"),
        # SVC(kernel="rbf"),
        # DecisionTreeClassifier(),
        GaussianProcessClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(n_estimators=200, max_features=15),
        AdaBoostClassifier(),
        XGBClassifier(),
        LogisticRegression(penalty='l1'),
        LogisticRegression(penalty='l2')
        ]


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

        ax = plt.subplot(3, len(classifiers)/3, plot_number)
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
    



def parameterSearch(data):

    Y = data['class'].values
    data = data.drop(['class'], axis = 1)
    X = data.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Adaboost parameterSearch

    parameters = {'base_estimator__splitter':["best", "random"],
                  'base_estimator__min_samples_leaf': [1, 5, 10], 
                  'base_estimator__max_depth': [5, 10, 30],
                  'base_estimator__max_features':[10, 15, 40],
                  'n_estimators': [50, 100, 200]
                  }

    adaBoost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
    clf = GridSearchCV(adaBoost, param_grid=parameters)
    clf.fit(X_train, Y_train)
    print (clf.best_estimator_)
    print (clf.best_params_)
    print (clf.best_score_)




def main():
    data = data_processing('~/Desktop/spotify_crawler/full_features/2017.csv')
    classiferCompare(data)
    # testFunction(data)
    # score = train_model(data)
    # print ("Model score:", score)

    # Parameter Search for Adaboost
    # parameterSearch(data)


if __name__ == "__main__":

    main()

