import sys; sys.path.insert(0, "../Attention_RI")
import numpy as np
import sklearn.metrics.pairwise as metrics
from dictionary import *
from sklearn.neighbors import NearestNeighbors

## -- DESCRIPTION --------------------------------------
#
#   Performs the TOEFL test on the word space
#
## -----------------------------------------------------

## -- CONFIGURATION ------------------------------------

# Which words to load
word_space = LogTransformRiDictionary("/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-100000-2000-2", 10, False) #W2vDictionary("/home/tobiasnorlund/Embeddings/GoogleNews-vectors-negative300.bin")#

# Where the toefl file is located
toefl_path = "/home/tobiasnorlund/Datasets/TOEFL/toefl.txt"

## -----------------------------------------------------

toefl = open(toefl_path)
correct = 0
tot = 0

for line in toefl:
    line = line[0:-1] # remove \n
    line_split = line.split(" ")

    vecs = np.zeros((5,word_space.d), dtype='float32')
    for i in range(5):
        vec = word_space.get_word_vector(line_split[i])
        if vec is not None:
            vecs[i, :] = vec
        else:
            print "Word not in dictionary: " + line_split[i]


    test_vec = vecs[0,:]

    diist = np.dot(vecs[0,:], vecs[1,:]) / (np.linalg.norm(vecs[0,:])*np.linalg.norm(vecs[1,:]))
    dists = metrics.cosine_similarity(np.atleast_2d(vecs[0,:]), vecs[1:,:])

    alts = vecs[1:, :]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(alts)
    dist, ind = nbrs.kneighbors(test_vec.reshape(1, -1))

    tot += 1
    if ind == 0:
        correct += 1
        print "CORRECT\t" + line + " (" + str(dists) + ")"
    else:
        print "FAIL\t" + line + " (" + str(dists) + ")"

print "\nFINISHED: " + str(float(correct) * 100 / tot) + "% correct"