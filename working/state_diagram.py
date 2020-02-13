import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import buffer

steps = 1100
with open ('state_buffers/{}.buffercopy'.format(steps), 'rb') as f:
    buf = pickle.load(f, encoding='latin1')

states = buf.states[:steps]
print (len(states))

x = states[:, 2].reshape(-1, 100)
y = states[:, 3].reshape(-1, 100)

for xi, yi in zip(x, y):
    plt.scatter(xi, yi, c=np.linspace(1, 100, 100), cmap=plt.get_cmap('cool'))
plt.colorbar()
plt.savefig('{}_diagram.png'.format(steps))
