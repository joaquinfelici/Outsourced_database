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

#def get_range(array, start, end): # simulates the query
#    return [array[i] for i in range(len(array)) if array[i] >= start and array[i] <= end]

def get_volume(array, start, end): # simulates the query
    volume = 0
    for i in range(len(array)):
        if array[i] >= start and array[i] <= end:
            volume += 1
    return volume

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
    sum_b = 0
    for x in data:
        if counter < width:
            sum_b += x
            counter += 1
        else:
            result.append(sum_b)
            sum_b = x
            counter = 1
    result.append(sum_b)
    return np.array(result)

def make_predictions(target_col, nb_iterations, n, hist_samples, data_file, mla='GNB'):
    # read dataset from a file
    DATA, IS_CAT = read_adult_data(data_file)
    # obtain the target column
    target_attribute = get_column(DATA, target_col)
    # generate all possible values in target column domain
    possible_values = range(min(target_attribute), max(target_attribute)+1, 1)
    # generate all possible queries
    combinations = [[a,b] for a in possible_values for b in possible_values if(a <= b)]

    #generate labels
    Labels = []
    for value in possible_values:
        Labels.extend([value]*n)

    acc = 0.0
    if nb_iterations == -1:
        nb_iterations = len(target_attribute)
    # iterate for every row
    for target_row in range(0, nb_iterations):
    #    print ("(" + str(target_row) + ") Generating training data...")
        actual_value = target_attribute[target_row]
        training_data = []
        # iterate for every possible value that the target cell may take
        for value in possible_values:
            target_attribute[target_row] = value
            # generate n histograms for every possible value
            for j in range(0, n, 1):
                histogram = np.zeros(len(target_attribute)+1)
                #generate hist_samples samples
                for i in range(0, hist_samples): # create one histogram
                    pair = get_uniform_pair(1, combinations)
                    histogram[get_volume(target_attribute, pair[0], pair[1])] += 1
                training_data.append(histogram)
        target_attribute[target_row] = actual_value

        #generate an input for the classifier
        input = np.zeros(len(target_attribute)+1)
        for i in range(0, hist_samples):
            pair = get_uniform_pair(1, combinations)
            input[get_volume(target_attribute, pair[0], pair[1])] += 1

        #train a classifier
      #  print "Training classifier..."
        clf = train_classifier(training_data, np.array(Labels), mla)  # mla specifies training algorithm to use
        #make a prediction
        predicted_value = clf.predict([input]).tolist()[0]
        if actual_value == predicted_value:
            acc += 1.0
       #     print  bcolors.OKGREEN + "Prediction is correct" + bcolors.ENDC
       # else:
        #    print  bcolors.FAIL + "Prediction is incorrect" + " (" + str(actual_value) + ", " + str(predicted_value) + ")" + bcolors.ENDC

    a = '%.2f' % (acc * 100 / float(nb_iterations))
    print (bcolors.BOLD + 'Accuracy was ' + a + '%' + bcolors.ENDC)
    return a


