from matplotlib import pyplot as plt
import numpy as np


a = np.load('ours.npy')

print(a)


plt.plot(np.arange(a.shape[0]),a)
plt.show()