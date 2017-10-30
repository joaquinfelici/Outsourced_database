"""
Test make_predictions method
"""

# !/usr/bin/env python
# coding=utf-8
from analyse import *

        
if __name__ == '__main__':



    # reduce_database(age_min, age_max, size, filename)
    reduce_database(20, 25, 100, "data/adult-short.all")

    target_column = 5  # target attribute position (5 is age)
    Nb_iterations = -1 # nb. of records to consider (-1 to consider all the records)
    N = 10          # number of histograms per label
    histogram_samples = 1000 # number of samples per histogram
    MLA = 'GNB'

    start_time = time.time()

    make_predictions(target_column, Nb_iterations, N, histogram_samples, MLA)

    elapsed_time = time.time() - start_time
    print ('Elapsed time ' + time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time)))
