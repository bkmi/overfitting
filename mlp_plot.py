import numpy as np
import matplotlib.pyplot as plt


fashion = np.load("fashion_mnist.npy").flatten()[:-900]
mnist = np.load("mnist.npy").flatten()[:-900]
fashion_corrupt = np.load("fashion_corrupt.npy").flatten()[:-400]
mnist_corrupt = np.load("mnist_corrupt.npy").flatten()[:-400]

fashion = [np.mean(i) for i in np.split(fashion, fashion.shape[0] // 1000)]
mnist = [np.mean(i) for i in np.split(mnist, mnist.shape[0] // 1000)]
fashion_corrupt = [np.mean(i) for i in np.split(fashion_corrupt, fashion_corrupt.shape[0] // 1000)]
mnist_corrupt = [np.mean(i) for i in np.split(mnist_corrupt, mnist_corrupt.shape[0] // 1000)]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fashion, label='Fashion MNIST')
ax.plot(mnist, label='MNIST')
ax.plot(fashion_corrupt, label='Fashion MNIST Corrupt')
ax.plot(mnist_corrupt, label='MNIST Corrupt')

ax.set_xlabel('Thousand Steps')
ax.set_ylabel('Average Loss')
ax.legend()

fig.tight_layout()
plt.savefig('corruption.png')
plt.show()
