import sys; sys.path.insert(0, "../Attention_RI")
import numpy as np
from dictionary import *
import matplotlib.pyplot as plt

## -- DESCRIPTION --------------------------------------
#
#   Plots the distribution of the words in a word space
#
## -----------------------------------------------------

## -- CONFIGURATION ------------------------------------

# Which words to load
word_space = RiDictionary("/media/tobiasnorlund/ac861917-9ad7-4905-93e9-ee6ab16360ad/bigdata/Dump/Wikipedia-100000-2000-2")#W2vDictionary("/home/tobiasnorlund/Embeddings/GoogleNews-vectors-negative300.bin")#

## -----------------------------------------------------

abs_vals = []
count_bins = np.zeros(50)
bins = np.logspace(0, 7, 50, base=10)

for (word, vec) in word_space.iter_words():
    abs_vals.append(np.linalg.norm(vec))

    if isinstance(word_space, RiDictionary):
        count = word_space.get_word_meta(word).focus_count
        for i in range(50):
            if count < bins[i]:
                count_bins[i] += 1
                break


n, bins, patches = plt.hist(abs_vals, bins=bins)

cm = plt.cm.get_cmap('YlOrRd')
i = 0
for p in patches:
    plt.setp(p, 'facecolor', cm(count_bins[i]/max(count_bins)))
    i += 1

plt.xscale('log')
plt.show()