import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


with open(sys.argv[1]) as s:
    data = json.load(s)

categories_counter = {}

for joke in data['jokes']:
    for category in joke['categories']:
        if category not in categories_counter:
            categories_counter[category] = 1
        else:
            categories_counter[category] += 1

ordered = OrderedDict(sorted(categories_counter.items(), reverse=True, key=lambda kv: kv[1]))
print(len(categories_counter))

plt.title('unijokes raw dataset categories (size {})'.format(len(data['jokes'])))
plt.bar(range(len(categories_counter)), sorted(categories_counter.values(), reverse=True), align='center', alpha=0.5)
plt.xticks(range(10), ordered.keys(), rotation='vertical', )
plt.show()
