from utils.read_data import read_adult_data
from utils.methods import reduce_database
from utils.methods import train_classifier
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

def get_uniform_pair(option, combinations =[]): # select 0 to generate both values independently
    if option == 0:           # other to uniformly select one pair from an array with all possible combinations
        a = int(np.random.uniform(lower, higher))
        b = int(np.random.uniform(lower, higher))
        return [min(a,b), max(a,b)]
    else:
        a = int(np.random.uniform(0, len(combinations)))
        return combinations[a]

def bucketize(data, width):
    """
    Takes a list and buketize each $width elements into one bucket
    by adding up all the related values.
    ex: buketize([1,4,3,6], 2) results in [5,9]
    :param data: a list of integers
    :param width: the width of each bucket
    :return: resulting list
    """
    result = []
    counter = 0
    summ = 0
    for x in data:
        if counter < width:
            summ += x
            counter += 1
        else:
            result.append(summ)
            summ = x
            counter = 1
    result.append(summ)
    return np.array(result)


def make_predictions(target_col, nb_iterations, n, hist_samples, b_width, mla='GNB'):

    # read dataset from a file
    DATA, IS_CAT = read_adult_data()

    target_attribute = get_column(DATA, target_col) # the column itself
    lower = min(target_attribute)    # minimum value for target attribute
    higher = max(target_attribute)   # maximum value for target attribute
    target_values = range(lower, higher+1, 1)
    combinations = [[a,b] for a in target_values for b in target_values if(a <= b)]

    acc = 0.0
    if nb_iterations == -1:
        nb_iterations = len(target_attribute)
        print nb_iterations
    for target_row in range(0, nb_iterations, 1): # len(target_attribute)):
        print ("(" + str(target_row) + ") Generating training data...")
        actual_value = target_attribute[target_row]
        training_data = []
        Labels = []
        for label in range (lower, higher + 1, 1): # create n histograms for each label
            target_attribute[target_row] = label
            for j in range(0, n, 1): # create n histograms for label k
                histogram = np.zeros(len(target_attribute)+1)
                for i in range(0, hist_samples, 1): # create one histogram
                    pair = get_uniform_pair(1, combinations)
                    histogram[len(get_range(target_attribute, pair[0], pair[1]))] += 1
                    # bucketize into b_width buckets
                    h = bucketize(histogram, b_width)
                    training_data.append(h)
                    Labels.append(label)
        #restore the original database
        target_attribute[target_row] = actual_value

        # generate input
        input = np.zeros(len(target_attribute)+1)
        for i in range(0, hist_samples, 1): # create one histogram
            pair = get_uniform_pair(1, combinations)
            input[len(get_range(target_attribute, pair[0], pair[1]))] += 1

        print "Training classifier..."
        print len(training_data[0])

        clf = train_classifier(training_data, np.array(Labels), mla)  # mla specifies training algorithm to use

        # predict
        predicted_value = clf.predict([input]).tolist()[0]

        if actual_value == predicted_value:
            acc += 1.0
            print  bcolors.OKGREEN + "Prediction is correct" + bcolors.ENDC
        else:
            print  bcolors.FAIL + "Prediction is incorrect" + " (" + str(actual_value) + ", " + str(predicted_value) + ")" + bcolors.ENDC

    a = '%.2f'%(acc*100/float(nb_iterations))
    print (bcolors.BOLD + 'Accuracy was ' + a + '%' + bcolors.ENDC)
    return a


