import numpy as np
import matplotlib.pyplot as plt

acc = np.load('Acc_2.npy')
x = np.arange(1,41)
plt.plot(x,acc)
plt.show()

loss = np.load('Loss_2.npy')
plt.plot(x,loss)
plt.show()