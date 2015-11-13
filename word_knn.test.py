import sys; sys.path.insert(0, "../Attention_RI")

from sklearn.neighbors import NearestNeighbors
from dictionary import *

## -- DESCRIPTION --------------------------------------
#
#   Looks up k nearest neighbors to words in a word space
#
## -----------------------------------------------------

## -- CONFIGURATION ------------------------------------

# Which words to load
word_space = RiDictionary("/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-100000-2000-2")

# How many neighbours to find
k = 10

## -----------------------------------------------------

sys.stdout.write("Loads all word vectors...")

(word_vectors, word_map) = word_space.get_all_word_vectors()

sys.stdout.write("\rBuilds model...")

nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine').fit(word_vectors)
sys.stdout.write("\r")
print "Enter the word to lookup:"

while True:

    test_word = raw_input()
    test_vec = word_space.get_word_vector(test_word)
    if test_vec is None:
        print "Not in dictionary, try again..."
        continue

    # Fetch neighbours
    distances, indices = nbrs.kneighbors(test_vec.reshape(1, -1))
    for i in range(10):
        print word_map.items()[indices[0][i]][0]