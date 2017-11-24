"""
Test make_predictions method
"""

# !/usr/bin/env python
# coding=utf-8
from analyse import *


MLA = 'GNB'
NB_histogram = [10]
NB_samples = [10**3]

c_TIME = dict()
ACCURACY = dict()

if __name__ == '__main__':

    age_min = 20
    age_max = 25
    size = 1000
    #reduce_database(age_min, age_max, size, "data/adult-short.all")

    target_column = 5  # target attribute position (5 is age)
    Nb_iterations = 5 # nb. of records to consider (-1 to consider all records)
    #bucket_width = 1 # width used to bucketize features (histogram)

    #N = 10          # number of histograms per label
    #histogram_samples = 1000 # number of samples per histogram

    # filename setup (old content are erased)
    filename = "results_%d_%d_%d_%d_%s"%(age_min, age_max, size, Nb_iterations, MLA)
    f = open(filename, 'w')
    f.write('Nb. Hist.\tNb. Samples\tAccuracy\tTime\n')
    f.close()

    for N in NB_histogram:
        for histogram_samples in NB_samples:
            start_time = time.time()
            accuracy = make_predictions(target_column, Nb_iterations, N, histogram_samples, MLA)
            ACCURACY['%d_%d'%(N, histogram_samples)] = accuracy
            elapsed_time = time.time() - start_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            c_TIME['%d_%d'%(N, histogram_samples)] = '%d:%02d:%02d'%(h, m, s)

            # append data to filename
            f = open(filename, 'a')
            f.write('%d\t%d\t%s\t%s\n'%(N, histogram_samples, ACCURACY['%d_%d'%(N, histogram_samples)], c_TIME['%d_%d'%(N, histogram_samples)]))
            f.close()


