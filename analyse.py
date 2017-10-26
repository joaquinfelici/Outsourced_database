from utils.read_data import read_adult_data
from utils.reduce_database import reduce_database
from utils.methods import *
import numpy as np
import time
import copy

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_column(matrix, i): # returns only one colum of the matrix
    try:
        return [int(row[i]) for row in matrix]
    except TypeError as e:
        return [row[i] for row in matrix]

def get_range(array, start, end): # simulates the query
    return [array[i] for i in range(len(array)) if array[i] >= start and array[i] <= end]

def get_uniform_pair(option): # select 0 to generate both values independently
    if option == 0:           # other to uniformly select one pair from an array with all possible combinations
        a = int(np.random.uniform(lower, higher))
        b = int(np.random.uniform(lower, higher))
        return [min(a,b), max(a,b)]
    else:
        a = int(np.random.uniform(0, len(combinations)))
        return combinations[a]

if __name__ == '__main__':

    start_time = time.time()

    # reduce_database(age_min, age_max, size, filename)
    reduce_database(20, 90, 100, "data/adult-short.all")
    DATA, IS_CAT = read_adult_data()
 
    target_col = 5  # the number of the column we have information about (5 is age)
    n = 1           # number of histograms per label

    target_attribute = get_column(DATA, target_col) # the column itself
    combinations = [[a,b] for a in target_attribute for b in target_attribute if(a <= b)]

    lower = min(target_attribute)    # minimum range for age
    higher = max(target_attribute)   # maximum range for age
    width = higher - lower + 1

    # histogram_samples = width**3
    histogram_samples = 10000 # why width**3? it's pretty effective but it takes ages

    acc = 0
    Nb_records_O = []
    for target_row in range(0, 10, 1): # len(target_attribute)):
        print ("(" + str(target_row) + ") Generating data...")
        actual_value = target_attribute[target_row]  
        training_data = []
        for label in range (lower, higher + 1, 1): # create n histograms for each label
            target_attribute[target_row] = label  
            for j in range(0, n, 1): # create n histograms for label k     
                histogram = [0] * (len(target_attribute)+1)
                for i in range(0, histogram_samples, 1): # create one histogram
                    pair = get_uniform_pair(1) # important: parameter must be changed below to generate the input
                    histogram[len(get_range(target_attribute, pair[0], pair[1]))] += 1
                    training_data.append([histogram, label])
        
        #restore the original database
        target_attribute[target_row] = actual_value

        histogram = [0] * (len(target_attribute)+1)
        for i in range(0, histogram_samples, 1): # create one histogram
            pair = get_uniform_pair(1)
            histogram[len(get_range(target_attribute, pair[0], pair[1]))] += 1
        input = copy.deepcopy(histogram) # only one of n inputs is considered

        print "Training classifier..."
        
        # prepare training data
        train = np.array(get_column(training_data, 0))
        labels = np.array(get_column(training_data, 1))
        
        # train classifier
        clf = train_classifier(train, labels, mla = 'GNB')  # mla specifies training algorithm to use
        
        # predict
        predicted_value = clf.predict([input]).tolist()[0]
        
        if actual_value == predicted_value:
            acc += 1
            print  bcolors.OKGREEN + "Prediction is correct" + bcolors.ENDC
        else:
            print  bcolors.FAIL + "Prediction is incorrect" + " (" + str(actual_value) + ", " + str(predicted_value) + ")" + bcolors.ENDC
    
    elapsed_time = time.time() - start_time
    
    print (bcolors.BOLD + 'Accuracy was ' + str(int(acc/float(10)*100)) + '%' + bcolors.ENDC) # the 10 value should depend on the number of iterations!
    print ('Elapsed time ' + time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time)))


