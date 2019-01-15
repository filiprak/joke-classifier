import json
import re
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
from nltk.stem import *

import data_provider

data_provider.init_data_provider(ngrams=True)

X, Y = data_provider.get_data(all_data=True)

print(len(X), len(Y))

# plt.title('unijokes raw dataset words (number of different words {})'.format(len(words_counter)))
# plt.bar(range(len(words_counter)), sorted(words_counter.values(), reverse=True), align='center', alpha=0.5, color='orange')
# plt.xticks(range(10), ordered.keys(), rotation='vertical')
# plt.show()
