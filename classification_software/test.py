import numpy as np
rand_id = np.random.random()
weights = [100,5,3,100]
normed_weights= np.cumsum(weights)/np.sum(weights)
print(normed_weights)
print(rand_id)
for i in range(len(normed_weights)):
    if normed_weights[i] >= rand_id:
        print(i)
    else:
        print('somethign went wrong')
