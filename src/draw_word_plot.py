import json
import re
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
from nltk.stem import *


with open(sys.argv[1]) as s:
    data = json.load(s)

words_counter = {}

stemmer = snowball.SnowballStemmer(language='english')

for joke in data['jokes']:
    for word in re.findall(r'\w+', joke['joke']):
        word_s = stemmer.stem(word)
        if word_s not in words_counter:
            words_counter[word_s] = 1
        else:
            words_counter[word_s] += 1

ordered = OrderedDict(sorted(words_counter.items(), reverse=True, key=lambda kv: kv[1]))
print(len(words_counter))

plt.title('unijokes raw dataset words (number of different words {})'.format(len(words_counter)))
plt.bar(range(len(words_counter)), sorted(words_counter.values(), reverse=True), align='center', alpha=0.5, color='orange')
plt.xticks(range(10), ordered.keys(), rotation='vertical')
plt.show()
