import numpy as np

a  = np.array([[0, 1, 2], [1,1,1]])

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.figure()