import sys; sys.path.insert(0, "../Attention_RI")
import numpy as np
import scipy.stats
from dictionary import *

## -- DESCRIPTION --------------------------------------
#
#   Calculates the cosine distances between pairs of word vectors and performs a spearman rank correlation with human
#   annotated values
#
## -----------------------------------------------------

## -- CONFIGURATION ------------------------------------

# Which words to load
word_space = RiDictionary("/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-100000-2000-2")#, 10, False) #W2vDictionary("/home/tobiasnorlund/Embeddings/GoogleNews-vectors-negative300.bin")#

# Where the toefl file is located
simlex_path = "/home/tobiasnorlund/Datasets/SimLex/simlex.txt"

## -----------------------------------------------------

def load_dataset(simlex_path):

    """
    Loads the SimLex dataset
    """

    f = open(simlex_path)

    X = []
    Y = []
    for line in f:
        line_split = line.rstrip('\n').split()
        X.append(line_split[0] + " " + line_split[1])
        Y.append(float(line_split[2])/10)

    return (X,Y)

def calc_dist(word1, word2):
    vec1 = word_space.get_word_vector(word1)
    vec2 = word_space.get_word_vector(word2)

    if vec1 is None or vec2 is None:
        return 0

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

# Load simlex dataset
(input_docs, Y) = load_dataset(simlex_path)

dist = []
for doc in input_docs:
    splitted = doc.split()
    dist.append(calc_dist(splitted[0], splitted[1]))

print "Spearman correlation on SimLex - result:"
print scipy.stats.spearmanr(dist, Y)