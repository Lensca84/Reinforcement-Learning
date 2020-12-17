import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Load model
try:
    model = torch.load('the_best_pb1.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)

nb_elem = 100

y = np.linspace(0, 1.5, nb_elem)
w = np.linspace(-np.pi, np.pi, nb_elem)

states = np.zeros((nb_elem**2, 8))

for i, y_i in enumerate(y):
    for j, w_j in enumerate(w):
        states[i*nb_elem + j][1] = y_i
        states[i*nb_elem + j][4] = w_j


states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
V = model(states_tensor).max(1)[0].detach().numpy()
VV = V.reshape(nb_elem, nb_elem)

YY, WW = np.meshgrid(y, w)

## Plot
plt.close()
fig = plt.figure(0)

ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(YY, WW, VV, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('Height of the lander')
ax.set_ylabel('Angle of the lander')
ax.set_zlabel('Estimated Value Function')

plt.show()
