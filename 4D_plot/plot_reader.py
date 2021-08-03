import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import numpy as np

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# ax.set_zlim(-1.01, 1.01)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter('{x:.02f}')

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

import csv

file = input("File number: ")

with open(str(file) + ".csv", newline = '') as csvfile:
    rows = list(csv.reader(csvfile))

    for i in range(5):
        print(rows[i])

    rows = rows[5:]
    headers = [head.strip() for head in rows[0]]
    rows = rows[1:]

    datas = [dict(zip(headers, [float(elem) for elem in row])) for row in rows]

# fixed lam
lam_val = -3.141592653589793
fixed_lam = filter(lambda data: abs(data['lam'] - lam_val) < 0.05*np.pi, datas)
X, Y, Z = [], [], []
for data in fixed_lam:
    X.append(data['theta'])
    Y.append(data['phi'])
    Z.append(data['score'])

x = np.reshape(X, (21, 21))
y = np.reshape(Y, (21, 21))
z = np.reshape(Z, (21, 21))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
