from utils.read_data import read_adult_data
from utils.methods import *
import numpy as np
import time
import copy

def get_column(matrix, i): # returns only one colum of the matrix
    try:
        return [int(row[i]) for row in matrix]
    except TypeError as e:
        return [row[i] for row in matrix]

def get_range(array, start, end): # simulates the query
    return [array[i] for i in range(len(array)) if array[i] >= start and array[i] <= end]

if __name__ == '__main__':

    start_time = time.time()
    
    DATA, IS_CAT = read_adult_data()
 
    target_col = 5  # the number of the column we have information about (5 is age)
    n = 1           # number of histograms per label

    target_attribute = get_column(DATA, target_col) # the column itself
    lower = min(target_attribute)    # minimum range for age
    higher = max(target_attribute)   # maximum range for age
    #print lower   
    #print higher  
    acc = 0
    for target_row in range(100):#len(target_attribute)):
        actual_value = target_attribute[target_row]  
        print actual_value
        training_data = []
        for label in range (lower, higher + 1, 1): # create n histograms for each label
            target_attribute[target_row] = label  
            for j in range(0, n, 1): # create n histograms for label k     
                histogram = [0] * (len(target_attribute)+1)
                for i in range(0, 10000, 1): # create one histogram
                    a = int(np.random.uniform(lower, higher))
                    b = int(np.random.uniform(lower, higher))
                    histogram[len(get_range(target_attribute, min(a,b), max(a,b)))] += 1
                #exclude the histogram related to the actual value from the training data
                if label == actual_value:
                    input  = copy.deepcopy(histogram) #copy data
                else:
                    training_data.append([histogram, label])
        #restore the original database
        target_attribute[target_row] = actual_value
        
        #prepare training data
        train = np.array(get_column(training_data, 0))
        labels = np.array(get_column(training_data, 1))
        #train classifier
        clf = train_classifier(train, labels, mla = 'GNB')  #$mla specifies training algorithm to use
        #predict
        if actual_value == clf.predict([input]).tolist()[0]:
            acc += 1
        print clf.predict([input]).tolist()[0]
        print 'Prediction Distribution:',  clf.predict_proba([input]).tolist()[0]
    print "acc:", acc
    
    elapsed_time = time.time() - start_time
    print ('Elapsed time ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


