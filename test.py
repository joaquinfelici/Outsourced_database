"""
Test  read_adult_data() and train_classifier() methods. 
"""

# !/usr/bin/env python
# coding=utf-8
from utils.read_data import read_adult_data
from utils.methods import *
import numpy as np
import random 
import copy


def write_data(data,  filename):
    """
    Write data to filename.
    $data is list of lists.
    """
    f = open(filename, 'w')
    for line in data:
        f.write(', '.join(str(x) for x in line) + '\n')
    f.close()
    return 0 
        
        
if __name__ == '__main__':
    
    #Read data from a file
    #IS_CAT specifies the type of the attributes (it is an additional information)
    DATA, IS_CAT = read_adult_data()
    #Print first record
    print DATA[0]
    
    #Randomlly select $size lines from data, if size < len(data).  
    selected_DATA = []
    length = len(DATA)
    size = 100
    temp_DATA = copy.deepcopy(DATA) #copy data
    if size <= length:
        counter = 0
        while(counter < size):
            i = random.choice(range(len(temp_DATA)))
            selected_DATA.append(temp_DATA.pop(i))
            counter += 1

    #Write selected data to a file
    write_data(selected_DATA, 'data.txt')

    #Generate some random data (1000 records) to train a classifier
    random_DATA = []
    for i in range(0, 1000, 1):
        record = []
        for i in range(0, 10, 1):
           rand = random.randint(0, 100)
           record.append(rand)
        random_DATA.append(record)
    #Train a classifier
    training_DATA = np.array(random_DATA)
    Label = np.array(100*range(10))
    clf = train_classifier(training_DATA, Label, mla='GNB')  #$mla specifies training algorithm to use
    #Use the classifier to obtain a prediction, then a prediction distribution
    input = [20, 70, 90, 55, 67, 17, 83, 46, 39, 4]
    print 'Prediction:',  clf.predict([input]).tolist()[0]
    print 'Prediction Distribution:',  clf.predict_proba([input]).tolist()[0]
