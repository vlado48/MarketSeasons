import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 50)
h = t[1]-t[0]
y_ex = np.empty(t.size)
y_im = np.empty(t.size)
y_ex[0] = y_im[0]  = 1 # initial conditions


for n, T in enumerate(t[:-1]):
    y_ex[n+1] = y_ex[n] + h * (-y_ex[n]**2)
    y_im[n+1] = y_im[n] / (1 + h * y_im[n])

plt.plot(t, y_ex, '-+', label='Explicit method')
plt.plot(t, y_im, '-o', label='Implicit method')
plt.plot(t, 1/(1+t), label='Exact solution')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.show()