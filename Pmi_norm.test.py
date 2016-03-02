import sys; sys.path.insert(0, "../Attention_RI/embedding/")
import numpy as np
import os
import struct
import pickle
from dictionary import *
from scipy.sparse import csr_matrix, lil_matrix

## -- DESCRIPTION --------------------------------------
#
#   Prints the Mutal information calculated while creating the PmiRi vectors
#
## -----------------------------------------------------

filepathprefix = "/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-2000000-2000-2"
word_space = RiDictionary(filepathprefix, normalize=False)#W2vDictionary("/home/tobiasnorlund/Embeddings/GoogleNews-vectors-negative300.bin")#
epsilon = 10


# Build sparse index vector matrix
print "Loads index vectors...\n"
if os.path.isfile(filepathprefix + ".index.pkl"):
    (R, context_counts) = pickle.load(open(filepathprefix + ".index.pkl"))
else:
    f = open(filepathprefix + ".index.bin", mode="rb")
    f_map = open(filepathprefix + ".map")
    R = lil_matrix((word_space.n,word_space.d), dtype="int8")
    context_counts = np.empty(word_space.n, dtype="uint32")
    for i in range(word_space.n):
        for e in range(epsilon):
            val = struct.unpack("h", f.read(2))[0]
            idx = val >> 1
            R[i,idx] = 1 if val % 2 == 1 else -1

        counts_str = f_map.readline().split("\t")
        context_counts[i] = int(counts_str[2]) if int(counts_str[2]) > 0 else 1

        sys.stdout.write("\r" + str(i))

    R = csr_matrix(R)
    pickle.dump((R, context_counts) , open(filepathprefix + ".index.pkl", mode="w"))
    f.close()
    f_map.close()

sum_ctxs = np.sum(context_counts)

def get_word_vector(word):
    vec = word_space.get_word_vector(word)
    if vec is not None:
        bow = np.maximum(0, R.dot(vec) / epsilon)
        bow = bow / (bow.sum() / word_space.word_map[word].context_count) # we know the total context counts. improve count estimates so that bow.sum() == true total count
        bow = bow * sum_ctxs / (word_space.word_map[word].focus_count*context_counts)
        #bow[bow == -np.inf] = 0

        return bow
    else:
        return None


print "Enter the word to lookup:"
while True:
    input = raw_input()
    bow = get_word_vector(input)
    if bow is not None:
        print bow