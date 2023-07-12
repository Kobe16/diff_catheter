import numpy as np
import matplotlib.pyplot as plt
import torch


ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

a = torch.tensor([5, 3, 2])
torch.Tensor.ndim = property(lambda self: len(self.shape))

# ax.plot(x, y, z, label='parametric curve')
ax.plot(a, label='point')

ax.legend()

plt.show()