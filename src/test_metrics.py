import numpy as np
import sklearn.metrics as m


Y_true = np.array([[1,0,0], [0,1,0], [0,0,1]])
Y =np.array([[1,0,0], [0,0,1], [1,0,0]])



print(m.recall_score(Y_true, Y, average='weighted'))
print(m.accuracy_score(Y_true, Y))
