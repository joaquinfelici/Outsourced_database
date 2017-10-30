
# !/usr/bin/env python
# coding=utf-8
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC,  SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from utils.read_data import read_adult_data
import copy


def train_classifier(training_data, label, mla='BNB'):
    """
    Trains a ML classifier based on $mla using data $training_data. 
    One can add any other training algorithms.
    """
    #Train a classifier
    if mla == 'GNB':
        model = GaussianNB()
    elif mla == 'BNB': 
        model = BernoulliNB()
    elif mla == 'DTC':
        model = tree.DecisionTreeClassifier()
    elif mla == 'RFC':
        model = RandomForestClassifier(n_estimators=25)
    elif mla == 'lSVC':
        clf = LinearSVC()
        model = CalibratedClassifierCV(clf, cv=2, method='isotonic') 
    elif mla == 'SVC':
        clf = SVC(kernel='linear')
        model = CalibratedClassifierCV(clf, cv=2, method='isotonic') 
    else:
        print 'The algorithm %s is not supported.'%mla
        print 'The default algorithm BNB:BernoulliNB() will be used.'
        print 
        model = BernoulliNB()
    model.fit(training_data, label)
    return model


def reduce_database(min, max, size, filename):
    with open("data/adult.all") as infile, open(filename, "w") as outfile:
        collector = []
        counter = 0

        for line in infile:
            if (counter >= size):
                break
            elif (int(line.split(',')[0]) >= min and int(line.split(',')[0]) <= max):
                outfile.write(line)
                counter += 1
