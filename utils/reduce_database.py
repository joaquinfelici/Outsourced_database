from utils.read_data import read_adult_data
from utils.methods import *
import copy

def reduce_database(min, max, size, filename): 

	with open("data/adult.all") as infile, open(filename,"w") as outfile:
		collector = []
		counter = 0
		
		for line in infile:
			if(counter >= size):
				break
			elif (int(line.split(',')[0]) >= min and int(line.split(',')[0]) <= max):
				outfile.write(line)
				counter += 1
			
